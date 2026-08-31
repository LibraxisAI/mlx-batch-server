from __future__ import annotations

import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ....auth.dependency import verify_auth
from ....provenance import get_runtime_provenance
from ...mlx.model_types import get_model_path
from ...mlx.runtime_aliases import (
    get_runtime_aliases,
    register_runtime_alias,
    resolve_runtime_target,
)
from ...mlx.runtime_attachments import (
    attach_runtime_surface,
    clear_runtime_surface_attachments,
    get_attached_models,
    get_attached_runtime_targets,
    get_remaining_runtime_surfaces,
    list_runtime_surface_attachments,
    list_runtime_surface_attachments_by_runtime,
    release_runtime_surface,
)
from ...mlx.runtime_policy import endpoint_runtime_session
from ...mlx.wrapper_cache import serialize_runtime_key
from .models_service import ModelsService
from .schema import (
    Model,
    ModelAliasRequest,
    ModelAliasResponse,
    ModelDeletion,
    ModelList,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelUnloadRequest,
    ModelUnloadResponse,
)

router = APIRouter(tags=["models"])

# Lazy initialization to avoid scanning cache during module import
_models_service = None
_TASK_ALIASES = {
    "llm": "llm",
    "chat": "llm",
    "text": "llm",
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "reranker": "embeddings",
    "visual": "visual",
    "visual-embeddings": "visual",
    "vision-embeddings": "visual",
    "images": "images",
    "image": "images",
    "vision": "images",
    "image-generation": "images",
    "stt": "stt",
    "asr": "stt",
    "whisper": "stt",
    "tts": "tts",
    "speech": "tts",
}
_ALLOWED_TASKS = set(_TASK_ALIASES.values())


def get_models_service() -> ModelsService:
    """Get or create the models service singleton with lazy initialization."""
    global _models_service
    if _models_service is None:
        _models_service = ModelsService()
    return _models_service


def _normalize_task(task: str | None) -> str | None:
    if not task:
        return None
    task = task.strip().lower()
    return _TASK_ALIASES.get(task, task)


def _build_llm_runtime_contract() -> dict[str, Any]:
    from ....core.config import get_settings

    settings = get_settings()
    return {
        "product_residency": "single_model",
        "text": {
            "runtime": "mlx-vlm.language_model",
            "tool_capable": True,
            "batch_capable": settings.enable_batch_inference,
        },
        "multimodal": {
            "runtime": "mlx-vlm",
            "batch_capable": settings.vlm_batch_enabled,
            "stream_batch_capable": settings.vlm_stream_batch_enabled,
            "execution": "single_flight",
        },
        "notes": [
            "Text requests batch through the resident model language tower.",
            "Eligible single-image vision requests can micro-batch on the shared mlx-vlm runtime.",
            "Video and multi-image requests intentionally stay on the mlx-vlm single-flight lane.",
        ],
    }


def _get_process_rss_gb() -> float | None:
    """Return peak resident set size in GB across macOS/Linux semantics."""
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    try:
        return round(float(rss) / divisor, 2)
    except Exception:
        return None


def _get_runtime_memory_snapshot() -> dict[str, float | int | None]:
    """Build one consistent physical-memory snapshot for runtime endpoints.

    We expose the new explicit field names and the older short aliases so the
    operator/tooling surface stays compatible across the sibling runtime repos.
    """
    from ....utils.memory import get_mlx_memory_snapshot

    mlx_mem = get_mlx_memory_snapshot()
    process_rss_gb = _get_process_rss_gb()
    mlx_active_memory_gb = mlx_mem.get("mlx_active_memory_gb")
    mlx_cache_memory_gb = mlx_mem.get("mlx_cache_memory_gb")
    return {
        "process_rss_gb": process_rss_gb,
        "rss_gb": process_rss_gb,
        "mlx_active_memory_gb": mlx_active_memory_gb,
        "mlx_active_gb": mlx_active_memory_gb,
        "mlx_cache_memory_gb": mlx_cache_memory_gb,
        "mlx_cache_gb": mlx_cache_memory_gb,
        "pid": os.getpid(),
    }


def _snapshot_llm_runtime() -> dict[str, Any]:
    from ....batch.coordinator import get_loaded_batch_models
    from ....chat.mlx.wrapper_cache import normalize_model_id, wrapper_cache
    from ....vision.vlm_batch import get_loaded_vlm_batch_models

    runtime_keys = [
        serialize_runtime_key(key) for key in wrapper_cache.get_runtime_keys()
    ]
    surface_attachments = list_runtime_surface_attachments()
    surface_runtime_attachments = list_runtime_surface_attachments_by_runtime()
    wrapper_loaded = sorted(
        {normalize_model_id(model_id) for model_id in wrapper_cache.get_loaded_models()}
    )
    vlm_loaded = sorted(
        {
            normalize_model_id(model_id)
            for model_id in wrapper_cache.get_loaded_vlm_models()
        }
    )
    batch_loaded = sorted(
        {normalize_model_id(model_id) for model_id in get_loaded_batch_models()}
    )
    vlm_batch_loaded = sorted(
        {normalize_model_id(model_id) for model_id in get_loaded_vlm_batch_models()}
    )
    cache_info = wrapper_cache.get_cache_info()
    cache_info.setdefault("runtime_keys", runtime_keys)
    contract = _build_llm_runtime_contract()

    wrapper_loaded_set = set(wrapper_loaded)
    vlm_loaded_set = set(vlm_loaded)
    batch_loaded_set = set(batch_loaded)
    vlm_batch_loaded_set = set(vlm_batch_loaded)
    shared_wrapper_residency = sorted(wrapper_loaded_set | vlm_loaded_set)

    data = []
    for model_id in sorted(
        set(shared_wrapper_residency).union(batch_loaded_set, vlm_batch_loaded_set)
    ):
        multimodal_resident = model_id in vlm_loaded_set
        text_resident = (
            model_id in wrapper_loaded_set
            or model_id in batch_loaded_set
            or multimodal_resident
        )
        active_lanes = []
        if text_resident:
            active_lanes.append("text")
        if multimodal_resident:
            active_lanes.append("multimodal")

        backends = ["wrapper"] if (text_resident or multimodal_resident) else []

        data.append(
            {
                "id": model_id,
                "loaded": True,
                "task": "llm",
                "backends": backends,
                "attached_tasks": surface_attachments.get(model_id, []),
                "runtime": {
                    "product_residency": "single_model",
                    "active_lanes": active_lanes,
                    "text": {
                        "resident": text_resident,
                        "runtime": contract["text"]["runtime"],
                        "tool_capable": contract["text"]["tool_capable"],
                        "batch_capable": contract["text"]["batch_capable"],
                        "batch_resident": model_id in batch_loaded_set,
                    },
                    "multimodal": {
                        "resident": multimodal_resident,
                        "runtime": contract["multimodal"]["runtime"],
                        "batch_capable": contract["multimodal"]["batch_capable"],
                        "stream_batch_capable": contract["multimodal"][
                            "stream_batch_capable"
                        ],
                        "batch_resident": model_id in vlm_batch_loaded_set,
                        "execution": contract["multimodal"]["execution"],
                    },
                },
            }
        )

    return {
        "data": data,
        "loaded_models": [entry["id"] for entry in data],
        "coordinators": {
            "llm_batch": batch_loaded,
            "vlm_batch": vlm_batch_loaded,
        },
        "caches": {"wrapper": shared_wrapper_residency},
        "runtime_keys": runtime_keys,
        "surface_attachments": surface_attachments,
        "surface_runtime_attachments": surface_runtime_attachments,
        "cache_info": cache_info,
        "runtime_contract": contract,
    }


def _snapshot_process_residency(llm_runtime: dict[str, Any]) -> dict[str, Any]:
    """Report every heavyweight owner in this process without loading weights."""
    from ....aux_runtime import get_aux_runtime_snapshot
    from ....embeddings.embeddings_service import get_embeddings_service
    from ....images.image_runtime import get_image_runtime_snapshot
    from ....stt.whisper_model import get_loaded_whisper_models
    from ....tts.tts_service import TTSService

    auxiliary = get_aux_runtime_snapshot()
    image = get_image_runtime_snapshot()
    by_backend = {
        "wrapper": llm_runtime["caches"]["wrapper"],
        "batch": llm_runtime["coordinators"]["llm_batch"],
        "vlm_batch": llm_runtime["coordinators"]["vlm_batch"],
        "image": list(image["resident_models"]),
        "embeddings": sorted(get_embeddings_service().get_loaded_native_models()),
        "tts": TTSService.get_loaded_models(),
        "stt": get_loaded_whisper_models(),
    }
    loaded_models = sorted(
        {model_id for models in by_backend.values() for model_id in models}
    )
    return {
        "loaded_models": loaded_models,
        "loaded_models_count": len(loaded_models),
        "loaded_models_by_backend": by_backend,
        "image_runtime": image,
        "auxiliary_runtime": auxiliary,
    }


def _build_llm_cache_info() -> dict[str, Any]:
    runtime = _snapshot_llm_runtime()
    cache_info = dict(runtime["cache_info"])
    cache_info["loaded_models"] = runtime["loaded_models"]
    cache_info["loaded_models_count"] = len(runtime["loaded_models"])
    cache_info["loaded_models_by_backend"] = {
        "wrapper": runtime["caches"]["wrapper"],
        "batch": runtime["coordinators"]["llm_batch"],
        "vlm_batch": runtime["coordinators"]["vlm_batch"],
    }
    cache_info["runtime_keys"] = runtime["runtime_keys"]
    cache_info["surface_attachments"] = runtime["surface_attachments"]
    cache_info["runtime_contract"] = runtime["runtime_contract"]
    return cache_info


def _format_retained_runtime_message(
    *,
    label: str,
    model_id: str,
    remaining_surfaces: list[str] | tuple[str, ...],
) -> str:
    retained_by = ", ".join(remaining_surfaces)
    return (
        f"{label} model {model_id} detached successfully; shared runtime retained by "
        f"{retained_by}"
    )


def _exact_runtime_requested(
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> bool:
    """Return True when the operator targeted one exact runtime variant."""
    return adapter_path is not None or draft_model_id is not None


def _select_runtime_targets(
    surface: str,
    *,
    model_id: str,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> list[Any]:
    """Return surface attachments for one model, optionally narrowed to one runtime."""
    runtime_targets = get_attached_runtime_targets(surface, model_id=model_id)
    if not _exact_runtime_requested(
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    ):
        return runtime_targets

    requested_target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    return [target for target in runtime_targets if target == requested_target]


def _visual_surfaces_still_attached(model_id: str) -> bool:
    """Return True when any exact runtime for this canonical model still owns vision."""
    return bool(get_attached_runtime_targets("visual", model_id=model_id))


@lru_cache(maxsize=128)
def _load_model_config(model_id: str) -> dict | None:
    model_path = Path(model_id)
    if model_path.exists():
        if model_path.is_dir():
            config_path = model_path / "config.json"
        else:
            config_path = model_path if model_path.name == "config.json" else None
            if config_path is None:
                config_path = model_path.parent / "config.json"
        if config_path.exists():
            try:
                with config_path.open(encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return None

    try:
        config_path = get_model_path(model_id) / "config.json"
        with config_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _is_llm_config(config: dict) -> bool:
    try:
        return get_models_service().scanner.is_model_supported(config)
    except Exception:
        return False


def _is_embeddings_config(config: dict) -> bool:
    model_type = config.get("model_type")
    if not model_type:
        return False
    model_type_norm = str(model_type).replace("-", "_").lower()
    if model_type_norm in {"qwen3_vl", "qwen3_vl_moe", "qwen2_vl", "qwen2_5_vl"}:
        return True
    if model_type_norm.startswith("qwen") and "vl" in model_type_norm:
        return True
    return (
        importlib.util.find_spec(f"mlx_embeddings.models.{model_type_norm}") is not None
    )


def _is_tts_config(config: dict) -> bool:
    model_type = config.get("model_type")
    if not model_type:
        return False
    try:
        from mlx_audio.tts.utils import MODEL_REMAPPING, get_available_models

        model_type_norm = MODEL_REMAPPING.get(model_type, model_type)
        return model_type_norm in get_available_models()
    except Exception:
        return False


def _detect_task_from_config(config: dict) -> str | None:
    task = None
    if _is_llm_config(config):
        task = "llm"
    else:
        model_type = str(config.get("model_type") or "").lower()
        if model_type == "whisper":
            task = "stt"
        elif _is_tts_config(config):
            task = "tts"
        elif _is_embeddings_config(config):
            task = "embeddings"
    return task


def _detect_task_from_name(model_id_lower: str) -> str | None:
    if "colqwen" in model_id_lower:
        return "visual"
    if "whisper" in model_id_lower or "asr" in model_id_lower:
        return "stt"
    if any(
        token in model_id_lower
        for token in (
            "tts",
            "kokoro",
            "vibevoice",
            "outetts",
            "chatterbox",
            "f5-tts",
            "f5tts",
            "dia",
            "csm",
            "sesame",
            "spark",
            "bark",
            "voxcpm",
            "indextts",
            "index-tts",
        )
    ):
        return "tts"
    if any(
        token in model_id_lower
        for token in ("flux", "mflux", "stable-diffusion", "sdxl", "sd3")
    ):
        return "images"
    if any(token in model_id_lower for token in ("embedding", "embed", "reranker")):
        return "embeddings"
    return None


def _detect_task(model_id: str, task_hint: str | None) -> str:
    task_hint = _normalize_task(task_hint)
    if task_hint:
        if task_hint not in _ALLOWED_TASKS:
            raise ValueError(
                f"Unknown task '{task_hint}'. Supported tasks: "
                + ", ".join(sorted(_ALLOWED_TASKS))
            )
        return task_hint

    task = None
    config = _load_model_config(model_id)
    if config:
        task = _detect_task_from_config(config)

    if task is None:
        task = _detect_task_from_name(model_id.lower())

    return task or "llm"


def extract_model_id_from_path(request: Request) -> str:
    """Extract full model ID from request path"""
    path = request.url.path
    prefix = "/v1/models/" if "/v1/models/" in path else "/models/"
    return path[len(prefix) :]


def handle_model_error(e: Exception) -> None:
    """Handle model-related errors and raise appropriate HTTP exceptions"""
    if isinstance(e, ValueError):
        raise HTTPException(status_code=404, detail=str(e))
    print(f"Error processing request: {e!s}")
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelList)
@router.get("/v1/models", response_model=ModelList)
async def list_models(
    include_details: bool = False,
    _auth: dict = Depends(verify_auth),
) -> ModelList:
    """List all available models"""
    return get_models_service().list_models(include_details)


@router.get("/models/loaded")
@router.get("/v1/models/loaded")
async def list_loaded_models(
    _auth: dict = Depends(verify_auth),
) -> dict:
    """
    List only models currently loaded in memory.

    Unlike /v1/models which lists all cached models on disk,
    this endpoint shows only models actively loaded in runtime caches.
    """
    runtime = _snapshot_llm_runtime()
    process_residency = _snapshot_process_residency(runtime)

    return {
        "object": "list",
        "data": runtime["data"],
        "loaded_models": process_residency["loaded_models"],
        "loaded_models_count": process_residency["loaded_models_count"],
        "loaded_models_by_backend": process_residency["loaded_models_by_backend"],
        "coordinators": runtime["coordinators"],
        "caches": runtime["caches"],
        "runtime_keys": runtime["runtime_keys"],
        "surface_attachments": runtime["surface_attachments"],
        "cache_info": runtime["cache_info"],
        "runtime_contract": runtime["runtime_contract"],
        "process_residency": process_residency,
        "runtime": _get_runtime_memory_snapshot(),
    }


@router.get("/models/{model_id:path}", response_model=Model)
@router.get("/v1/models/{model_id:path}", response_model=Model)
async def get_model(
    model_id: str,
    include_details: bool = False,
    _auth: dict = Depends(verify_auth),
) -> Model:
    """Get information about a specific model"""
    model = get_models_service().get_model(model_id, include_details)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/models/{model_id:path}", response_model=ModelDeletion)
@router.delete("/v1/models/{model_id:path}", response_model=ModelDeletion)
async def delete_model(
    request: Request,
    _auth: dict = Depends(verify_auth),
) -> ModelDeletion:
    """
    Delete a fine-tuned model from local cache.
    """
    try:
        model_id = extract_model_id_from_path(request)
        return get_models_service().delete_model(model_id)
    except Exception as e:
        handle_model_error(e)


async def _load_llm_model(request: ModelLoadRequest) -> ModelLoadResponse:
    """Load an LLM model into memory with optional alias and cache eviction."""
    runtime_target = resolve_runtime_target(
        request.model,
        adapter_path=request.adapter_path,
        draft_model_id=request.draft_model_id,
    )
    canonical_model_id = runtime_target.model_id
    alias_message = ""
    if request.alias:
        canonical_model_id = register_runtime_alias(
            request.alias,
            canonical_model_id,
            adapter_path=runtime_target.adapter_path,
            draft_model_id=runtime_target.draft_model_id,
        )
        alias_message = (
            f" (runtime alias registered: {request.alias} -> {canonical_model_id})"
        )

    async with endpoint_runtime_session(
        model_id=canonical_model_id,
        adapter_path=runtime_target.adapter_path,
        draft_model_id=runtime_target.draft_model_id,
    ) as switch_result:
        result = get_models_service().load_model(
            model_id=canonical_model_id,
            adapter_path=runtime_target.adapter_path,
            draft_model_id=runtime_target.draft_model_id,
        )
        attach_runtime_surface(
            canonical_model_id,
            "llm",
            adapter_path=runtime_target.adapter_path,
            draft_model_id=runtime_target.draft_model_id,
        )

    result["id"] = canonical_model_id
    result["task"] = "llm"
    if alias_message:
        result["message"] = result["message"] + alias_message
    if switch_result["switched"]:
        evicted = switch_result["evicted_models"]
        suffix = f" after evicting {len(evicted)} prior model(s): " + ", ".join(evicted)
        result["message"] = result["message"] + suffix
    result["cache_info"] = _build_llm_cache_info()
    return ModelLoadResponse(**result)


@router.post("/models/load", response_model=ModelLoadResponse)
@router.post("/v1/models/load", response_model=ModelLoadResponse)
async def load_model(  # noqa: PLR0911, PLR0912, PLR0915
    request: ModelLoadRequest,
    _auth: dict = Depends(verify_auth),
) -> ModelLoadResponse:
    """
    Load a model into memory for inference.

    This endpoint loads an MLX model into the cache, making it ready for
    inference requests. If the model is already loaded, returns success
    with 'already_loaded' status (idempotent).

    The model will be automatically unloaded after the TTL expires (default: 10 min)
    or when cache capacity is reached (LRU eviction).
    """
    try:
        task = _detect_task(request.model, request.task)
        if task == "llm":
            return await _load_llm_model(request)

        if task == "embeddings":
            from ....embeddings.embeddings_service import get_embeddings_service

            service = get_embeddings_service()
            if service.uses_shared_vlm_runtime(request.model):
                runtime_target = resolve_runtime_target(
                    request.model,
                    adapter_path=request.adapter_path,
                    draft_model_id=request.draft_model_id,
                )
                canonical_model_id = runtime_target.model_id
                alias_message = ""
                if request.alias:
                    canonical_model_id = register_runtime_alias(
                        request.alias,
                        canonical_model_id,
                        adapter_path=runtime_target.adapter_path,
                        draft_model_id=runtime_target.draft_model_id,
                    )
                    alias_message = (
                        f" (runtime alias registered: {request.alias} -> "
                        f"{canonical_model_id})"
                    )

                async with endpoint_runtime_session(
                    model_id=canonical_model_id,
                    adapter_path=runtime_target.adapter_path,
                    draft_model_id=runtime_target.draft_model_id,
                ) as switch_result:
                    already_loaded = service.load_model(
                        canonical_model_id,
                        adapter_path=runtime_target.adapter_path,
                        draft_model_id=runtime_target.draft_model_id,
                    )
                    attach_runtime_surface(
                        canonical_model_id,
                        "embeddings",
                        adapter_path=runtime_target.adapter_path,
                        draft_model_id=runtime_target.draft_model_id,
                    )

                message = (
                    f"Embeddings model {canonical_model_id} was already loaded"
                    if already_loaded
                    else f"Embeddings model {canonical_model_id} loaded successfully"
                )
                if alias_message:
                    message = message + alias_message
                if switch_result["switched"]:
                    evicted = switch_result["evicted_models"]
                    message = (
                        message
                        + f" after evicting {len(evicted)} prior model(s): "
                        + ", ".join(evicted)
                    )

                return ModelLoadResponse(
                    id=canonical_model_id,
                    task="embeddings",
                    status="already_loaded" if already_loaded else "loaded",
                    message=message,
                    cache_info=_build_llm_cache_info(),
                )

            from ....aux_runtime import auxiliary_runtime_operation

            canonical_model_id = service.canonicalize_model_id(request.model)
            with auxiliary_runtime_operation("embeddings", canonical_model_id):
                already_loaded = service.load_model(request.model)
            return ModelLoadResponse(
                id=request.model,
                task="embeddings",
                status="already_loaded" if already_loaded else "loaded",
                message=(
                    f"Embeddings model {request.model} was already loaded"
                    if already_loaded
                    else f"Embeddings model {request.model} loaded successfully"
                ),
                cache_info=None,
            )

        if task == "visual":
            from ....chat.mlx.wrapper_cache import normalize_model_id, wrapper_cache
            from ....embeddings.visual_router import (
                get_visual_embedder,
                has_visual_embedder,
            )

            runtime_target = resolve_runtime_target(
                request.model,
                adapter_path=request.adapter_path,
                draft_model_id=request.draft_model_id,
            )
            canonical_model_id = normalize_model_id(runtime_target.model_id)
            alias_message = ""
            if request.alias:
                canonical_model_id = register_runtime_alias(
                    request.alias,
                    canonical_model_id,
                    adapter_path=runtime_target.adapter_path,
                    draft_model_id=runtime_target.draft_model_id,
                )
                alias_message = (
                    f" (runtime alias registered: {request.alias} -> "
                    f"{canonical_model_id})"
                )

            already_loaded = has_visual_embedder(
                canonical_model_id,
                adapter_path=runtime_target.adapter_path,
                draft_model_id=runtime_target.draft_model_id,
            ) or wrapper_cache.is_runtime_loaded(
                canonical_model_id,
                adapter_path=runtime_target.adapter_path,
                draft_model_id=runtime_target.draft_model_id,
            )
            async with endpoint_runtime_session(
                model_id=canonical_model_id,
                adapter_path=runtime_target.adapter_path,
                draft_model_id=runtime_target.draft_model_id,
            ) as switch_result:
                get_visual_embedder(
                    canonical_model_id,
                    adapter_path=runtime_target.adapter_path,
                    draft_model_id=runtime_target.draft_model_id,
                )
                attach_runtime_surface(
                    canonical_model_id,
                    "visual",
                    adapter_path=runtime_target.adapter_path,
                    draft_model_id=runtime_target.draft_model_id,
                )

            message = (
                f"Visual embeddings model {canonical_model_id} was already loaded"
                if already_loaded
                else f"Visual embeddings model {canonical_model_id} loaded successfully"
            )
            if alias_message:
                message = message + alias_message
            if switch_result["switched"]:
                evicted = switch_result["evicted_models"]
                message = (
                    message
                    + f" after evicting {len(evicted)} prior model(s): "
                    + ", ".join(evicted)
                )

            return ModelLoadResponse(
                id=canonical_model_id,
                task="visual",
                status="already_loaded" if already_loaded else "loaded",
                message=message,
                cache_info=_build_llm_cache_info(),
            )

        if task == "stt":
            from ....aux_runtime import auxiliary_runtime_operation
            from ....stt.whisper_model import preload_whisper_model

            with auxiliary_runtime_operation("stt", request.model):
                already_loaded = preload_whisper_model(request.model)
            return ModelLoadResponse(
                id=request.model,
                task="stt",
                status="already_loaded" if already_loaded else "loaded",
                message=(
                    f"STT model {request.model} was already loaded"
                    if already_loaded
                    else f"STT model {request.model} loaded successfully"
                ),
                cache_info=None,
            )

        if task == "tts":
            from ....aux_runtime import auxiliary_runtime_operation
            from ....tts.tts_service import TTSService

            with auxiliary_runtime_operation("tts", request.model):
                already_loaded = TTSService.preload_model(request.model)
            return ModelLoadResponse(
                id=request.model,
                task="tts",
                status="already_loaded" if already_loaded else "loaded",
                message=(
                    f"TTS model {request.model} was already loaded"
                    if already_loaded
                    else f"TTS model {request.model} loaded successfully"
                ),
                cache_info=None,
            )

        if task == "images":
            model_lower = request.model.lower()
            if "stable-diffusion" in model_lower or "sdxl" in model_lower:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Stable Diffusion models are not supported by the image "
                        "backend. Use an mflux/FLUX model or add an SD backend."
                    ),
                )

            from ....images.image_runtime import get_image_runtime_pool

            already_loaded = await get_image_runtime_pool().prewarm(request.model)
            return ModelLoadResponse(
                id=request.model,
                task="images",
                status="already_loaded" if already_loaded else "loaded",
                message=(
                    f"Image model {request.model} was already loaded"
                    if already_loaded
                    else f"Image model {request.model} loaded successfully"
                ),
                cache_info=None,
            )

        raise HTTPException(status_code=400, detail=f"Unsupported task '{task}'")
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/aliases")
@router.get("/v1/models/aliases")
async def list_model_aliases(
    _auth: dict = Depends(verify_auth),
) -> dict[str, Any]:
    """List in-process runtime aliases available to operator tooling."""
    aliases = get_runtime_aliases()
    return {
        "object": "list",
        "data": [
            {
                "alias": alias,
                "model": model_id,
            }
            for alias, model_id in sorted(aliases.items())
        ],
    }


@router.post("/models/alias", response_model=ModelAliasResponse)
@router.post("/v1/models/alias", response_model=ModelAliasResponse)
async def create_model_alias(
    request: ModelAliasRequest,
    _auth: dict = Depends(verify_auth),
) -> ModelAliasResponse:
    """Register an alias for an existing or future runtime target."""
    try:
        target = resolve_runtime_target(
            request.model,
            adapter_path=request.adapter_path,
            draft_model_id=request.draft_model_id,
        )
        canonical_model_id = register_runtime_alias(
            request.alias,
            target.model_id,
            adapter_path=target.adapter_path,
            draft_model_id=target.draft_model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ModelAliasResponse(
        alias=request.alias,
        model=canonical_model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
    )


def _build_unload_response(
    *,
    task: str | None,
    status: str,
    message: str,
    unloaded_models: list[str],
) -> ModelUnloadResponse:
    return ModelUnloadResponse(
        task=task,
        status=status,
        message=message,
        unloaded_models=unloaded_models,
        cache_info=None,
    )


def _unload_vlm_runtime(
    model_id: str | None = None,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> list[str]:
    """Route VLM unloads through the compatibility seam backed by wrapper_cache."""
    from ....vision.vlm_cache import unload_vlm_model

    unload_kwargs: dict[str, str] = {}
    if adapter_path is not None:
        unload_kwargs["adapter_path"] = adapter_path
    if draft_model_id is not None:
        unload_kwargs["draft_model_id"] = draft_model_id
    return unload_vlm_model(model_id, **unload_kwargs)


async def _unload_shared_embeddings_surface(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> ModelUnloadResponse:
    from ....embeddings.embeddings_service import get_embeddings_service
    from ....vision.vlm_batch import shutdown_vlm_coordinator

    service = get_embeddings_service()
    shared_runtime = service.uses_shared_vlm_runtime(model_id)
    runtime_target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    canonical_model_id = (
        service.canonicalize_model_id(runtime_target.model_id)
        if shared_runtime
        else runtime_target.model_id
    )
    runtime_targets = _select_runtime_targets(
        "embeddings",
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    if not runtime_targets:
        unloaded = (
            [canonical_model_id]
            if service.unload_model(
                runtime_target.model_id,
                adapter_path=runtime_target.adapter_path,
                draft_model_id=runtime_target.draft_model_id,
                release_runtime=True,
            )
            else []
        )
        if unloaded and not _visual_surfaces_still_attached(canonical_model_id):
            await shutdown_vlm_coordinator(canonical_model_id)
        return ModelUnloadResponse(
            task="embeddings",
            status="unloaded" if unloaded else "not_found",
            message=(
                f"Embeddings model {canonical_model_id} unloaded successfully"
                if unloaded
                else f"Embeddings model {canonical_model_id} was not loaded"
            ),
            unloaded_models=unloaded,
            cache_info=_build_llm_cache_info() if shared_runtime else None,
        )

    unloaded: list[str] = []
    remaining_surfaces: set[str] = set()
    detached = False
    for target in runtime_targets:
        target_remaining = get_remaining_runtime_surfaces(
            target.model_id,
            releasing_surface="embeddings",
            adapter_path=target.adapter_path,
            draft_model_id=target.draft_model_id,
        )
        preserve_runtime = bool(target_remaining)
        unload_kwargs: dict[str, Any] = {"release_runtime": not preserve_runtime}
        if target.adapter_path is not None:
            unload_kwargs["adapter_path"] = target.adapter_path
        if target.draft_model_id is not None:
            unload_kwargs["draft_model_id"] = target.draft_model_id
        if service.unload_model(
            target.model_id,
            **unload_kwargs,
        ):
            unloaded.append(target.model_id)
            detached = detached or preserve_runtime
        release_runtime_surface(
            target.model_id,
            "embeddings",
            adapter_path=target.adapter_path,
            draft_model_id=target.draft_model_id,
        )
        remaining_surfaces.update(target_remaining)

    unloaded = list(dict.fromkeys(unloaded))
    if unloaded and not _visual_surfaces_still_attached(canonical_model_id):
        await shutdown_vlm_coordinator(canonical_model_id)
    if unloaded and detached:
        status = "detached"
        message = _format_retained_runtime_message(
            label="Embeddings",
            model_id=canonical_model_id,
            remaining_surfaces=sorted(remaining_surfaces),
        )
    else:
        status = "unloaded" if unloaded else "not_found"
        message = (
            f"Embeddings model {canonical_model_id} unloaded successfully"
            if unloaded
            else f"Embeddings model {canonical_model_id} was not loaded"
        )
    return ModelUnloadResponse(
        task="embeddings",
        status=status,
        message=message,
        unloaded_models=unloaded,
        cache_info=_build_llm_cache_info() if shared_runtime else None,
    )


async def _unload_visual_surface(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> ModelUnloadResponse:
    from ....embeddings.visual_router import unload_visual_embedder
    from ....vision.vlm_batch import shutdown_vlm_coordinator

    runtime_target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    runtime_targets = _select_runtime_targets(
        "visual",
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    canonical_model_id = runtime_target.model_id

    if not runtime_targets:
        unloaded = unload_visual_embedder(
            runtime_target.model_id,
            adapter_path=runtime_target.adapter_path,
            draft_model_id=runtime_target.draft_model_id,
            release_runtime=True,
        )
        if unloaded and not _visual_surfaces_still_attached(canonical_model_id):
            await shutdown_vlm_coordinator(canonical_model_id)
        return ModelUnloadResponse(
            task="visual",
            status="unloaded" if unloaded else "not_found",
            message=(
                f"Visual model {canonical_model_id} unloaded successfully"
                if unloaded
                else f"Visual model {canonical_model_id} was not loaded"
            ),
            unloaded_models=unloaded,
            cache_info=_build_llm_cache_info(),
        )

    unloaded: list[str] = []
    remaining_surfaces: set[str] = set()
    detached = False
    for target in runtime_targets:
        target_remaining = get_remaining_runtime_surfaces(
            target.model_id,
            releasing_surface="visual",
            adapter_path=target.adapter_path,
            draft_model_id=target.draft_model_id,
        )
        preserve_runtime = bool(target_remaining)
        unload_kwargs: dict[str, str] = {}
        if target.adapter_path is not None:
            unload_kwargs["adapter_path"] = target.adapter_path
        if target.draft_model_id is not None:
            unload_kwargs["draft_model_id"] = target.draft_model_id
        unloaded.extend(
            unload_visual_embedder(
                target.model_id,
                release_runtime=not preserve_runtime,
                **unload_kwargs,
            )
        )
        release_runtime_surface(
            target.model_id,
            "visual",
            adapter_path=target.adapter_path,
            draft_model_id=target.draft_model_id,
        )
        detached = detached or preserve_runtime
        remaining_surfaces.update(target_remaining)

    unloaded = list(dict.fromkeys(unloaded))
    if unloaded and not _visual_surfaces_still_attached(canonical_model_id):
        await shutdown_vlm_coordinator(canonical_model_id)
    if unloaded and detached:
        status = "detached"
        message = _format_retained_runtime_message(
            label="Visual",
            model_id=canonical_model_id,
            remaining_surfaces=sorted(remaining_surfaces),
        )
    else:
        status = "unloaded" if unloaded else "not_found"
        message = (
            f"Visual model {canonical_model_id} unloaded successfully"
            if unloaded
            else f"Visual model {canonical_model_id} was not loaded"
        )
    return ModelUnloadResponse(
        task="visual",
        status=status,
        message=message,
        unloaded_models=unloaded,
        cache_info=_build_llm_cache_info(),
    )


async def _unload_specific(  # noqa: PLR0911, PLR0912, PLR0915
    task: str,
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> ModelUnloadResponse:
    if task == "llm":
        from ....batch.coordinator import shutdown_batch_coordinator
        from ....vision.vlm_batch import shutdown_vlm_coordinator

        runtime_target = resolve_runtime_target(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        runtime_targets = _select_runtime_targets(
            "llm",
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        canonical_model_id = runtime_target.model_id
        await shutdown_batch_coordinator(
            canonical_model_id,
            adapter_path=(
                runtime_target.adapter_path
                if _exact_runtime_requested(
                    adapter_path=adapter_path,
                    draft_model_id=draft_model_id,
                )
                else None
            ),
        )

        if not runtime_targets:
            result = get_models_service().unload_model(
                model_id=canonical_model_id,
                adapter_path=runtime_target.adapter_path,
                draft_model_id=runtime_target.draft_model_id,
                release_runtime=True,
            )
            unloaded_models = list(dict.fromkeys(result.get("unloaded_models", [])))
            if unloaded_models and not _visual_surfaces_still_attached(
                canonical_model_id
            ):
                await shutdown_vlm_coordinator(canonical_model_id)
            status = "unloaded" if unloaded_models else result["status"]
            message = (
                f"Model {canonical_model_id} unloaded successfully"
                if unloaded_models
                else result["message"]
            )
            return ModelUnloadResponse(
                task="llm",
                status=status,
                message=message,
                unloaded_models=unloaded_models,
                cache_info=_build_llm_cache_info(),
            )

        unloaded_models: list[str] = []
        remaining_surfaces: set[str] = set()
        detached = False
        for target in runtime_targets:
            target_remaining = get_remaining_runtime_surfaces(
                target.model_id,
                releasing_surface="llm",
                adapter_path=target.adapter_path,
                draft_model_id=target.draft_model_id,
            )
            preserve_runtime = bool(target_remaining)
            unload_kwargs: dict[str, Any] = {
                "model_id": target.model_id,
                "release_runtime": not preserve_runtime,
            }
            if target.adapter_path is not None:
                unload_kwargs["adapter_path"] = target.adapter_path
            if target.draft_model_id is not None:
                unload_kwargs["draft_model_id"] = target.draft_model_id
            result = get_models_service().unload_model(**unload_kwargs)
            release_runtime_surface(
                target.model_id,
                "llm",
                adapter_path=target.adapter_path,
                draft_model_id=target.draft_model_id,
            )
            if result.get("unloaded_models"):
                unloaded_models.extend(result.get("unloaded_models", []))
                if not _visual_surfaces_still_attached(canonical_model_id):
                    await shutdown_vlm_coordinator(canonical_model_id)
            detached = detached or preserve_runtime
            remaining_surfaces.update(target_remaining)

        unloaded_models = list(dict.fromkeys(unloaded_models))

        status = "detached" if unloaded_models and detached else "unloaded"
        message = (
            _format_retained_runtime_message(
                label="LLM",
                model_id=canonical_model_id,
                remaining_surfaces=sorted(remaining_surfaces),
            )
            if unloaded_models and detached
            else (
                f"Model {canonical_model_id} unloaded successfully"
                if unloaded_models
                else f"Model {canonical_model_id} was not loaded"
            )
        )
        if not unloaded_models:
            status = "not_found"
        return ModelUnloadResponse(
            task="llm",
            status=status,
            message=message,
            unloaded_models=unloaded_models,
            cache_info=_build_llm_cache_info(),
        )

    if task == "embeddings":
        return await _unload_shared_embeddings_surface(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )

    if task == "visual":
        return await _unload_visual_surface(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )

    if task == "stt":
        from ....stt.whisper_model import unload_whisper_model

        unloaded = unload_whisper_model(model_id)
        status = "unloaded" if unloaded else "not_found"
        message = (
            f"STT model {model_id} unloaded successfully"
            if unloaded
            else f"STT model {model_id} was not loaded"
        )
        return _build_unload_response(
            task="stt",
            status=status,
            message=message,
            unloaded_models=unloaded,
        )

    if task == "tts":
        from ....tts.tts_service import TTSService

        unloaded = TTSService.unload_model(model_id)
        status = "unloaded" if unloaded else "not_found"
        message = (
            f"TTS model {model_id} unloaded successfully"
            if unloaded
            else f"TTS model {model_id} was not loaded"
        )
        return _build_unload_response(
            task="tts",
            status=status,
            message=message,
            unloaded_models=unloaded,
        )

    if task == "images":
        from ....images.image_runtime import get_image_runtime_pool

        unloaded = [model_id] if await get_image_runtime_pool().unload(model_id) else []
        status = "unloaded" if unloaded else "not_found"
        message = (
            f"Image model {model_id} unloaded successfully"
            if unloaded
            else f"Image model {model_id} was not loaded"
        )
        return _build_unload_response(
            task="images",
            status=status,
            message=message,
            unloaded_models=unloaded,
        )

    raise HTTPException(status_code=400, detail=f"Unsupported task '{task}'")


async def _clear_llm_task() -> ModelUnloadResponse:
    """Clear llm surfaces while preserving shared VLM runtime ownership."""
    from ....batch.coordinator import shutdown_all_coordinators
    from ....vision.vlm_batch import (
        shutdown_all_vlm_coordinators,
        shutdown_vlm_coordinator,
    )

    await shutdown_all_coordinators()
    llm_models = get_attached_models("llm")

    # Legacy fallback: older runtime state may have resident llm wrappers
    # without attachment bookkeeping yet. Preserve the old operator behavior
    # only when the whole surface registry is empty.
    if not llm_models and not list_runtime_surface_attachments():
        await shutdown_all_vlm_coordinators()
        result = get_models_service().unload_model(model_id=None)
        unloaded_models = list(result.get("unloaded_models", []))
        unloaded_models.extend(_unload_vlm_runtime())
        result["task"] = "llm"
        result["unloaded_models"] = list(dict.fromkeys(unloaded_models))
        result["message"] = (
            f"Cleared {len(result['unloaded_models'])} llm model(s); "
            "legacy runtime state had no surface attachments"
        )
        result["cache_info"] = _build_llm_cache_info()
        return ModelUnloadResponse(**result)

    unloaded_models: list[str] = []
    detached_models: list[str] = []
    for target in get_attached_runtime_targets("llm"):
        target_remaining = get_remaining_runtime_surfaces(
            target.model_id,
            releasing_surface="llm",
            adapter_path=target.adapter_path,
            draft_model_id=target.draft_model_id,
        )
        preserve_runtime = bool(target_remaining)
        unload_kwargs: dict[str, Any] = {
            "model_id": target.model_id,
            "release_runtime": not preserve_runtime,
        }
        if target.adapter_path is not None:
            unload_kwargs["adapter_path"] = target.adapter_path
        if target.draft_model_id is not None:
            unload_kwargs["draft_model_id"] = target.draft_model_id
        result = get_models_service().unload_model(**unload_kwargs)
        release_runtime_surface(
            target.model_id,
            "llm",
            adapter_path=target.adapter_path,
            draft_model_id=target.draft_model_id,
        )
        if result.get("unloaded_models"):
            unloaded_models.extend(result["unloaded_models"])
            if preserve_runtime:
                detached_models.extend(result["unloaded_models"])
            if not _visual_surfaces_still_attached(target.model_id):
                await shutdown_vlm_coordinator(target.model_id)

    unloaded_models = list(dict.fromkeys(unloaded_models))
    return ModelUnloadResponse(
        task="llm",
        status="cleared",
        unloaded_models=unloaded_models,
        message=(
            f"Cleared {len(unloaded_models)} llm model(s); "
            f"{len(detached_models)} shared runtime(s) stayed hot for other surfaces"
        ),
        cache_info=_build_llm_cache_info(),
    )


async def _clear_task(  # noqa: PLR0911, PLR0912, PLR0915
    task: str,
) -> ModelUnloadResponse:
    if task == "llm":
        return await _clear_llm_task()

    if task == "embeddings":
        from ....embeddings.embeddings_service import get_embeddings_service
        from ....vision.vlm_batch import shutdown_vlm_coordinator

        service = get_embeddings_service()
        shared_runtime = service.has_shared_vlm_runtime_models()
        unloaded = service.clear_native_models()
        shared_models = sorted(
            set(service.get_shared_vlm_models()).union(
                get_attached_models("embeddings")
            )
        )
        runtime_targets = get_attached_runtime_targets("embeddings")
        if not runtime_targets:
            for model_id in shared_models:
                if service.unload_model(model_id, release_runtime=True):
                    unloaded.append(model_id)
                    await shutdown_vlm_coordinator(model_id)
            unloaded = list(dict.fromkeys(unloaded))
            return ModelUnloadResponse(
                task="embeddings",
                status="cleared",
                message=f"Cleared {len(unloaded)} embeddings model(s) from cache",
                unloaded_models=unloaded,
                cache_info=_build_llm_cache_info() if shared_runtime else None,
            )

        for target in runtime_targets:
            target_remaining = get_remaining_runtime_surfaces(
                target.model_id,
                releasing_surface="embeddings",
                adapter_path=target.adapter_path,
                draft_model_id=target.draft_model_id,
            )
            preserve_runtime = bool(target_remaining)
            unload_kwargs: dict[str, Any] = {"release_runtime": not preserve_runtime}
            if target.adapter_path is not None:
                unload_kwargs["adapter_path"] = target.adapter_path
            if target.draft_model_id is not None:
                unload_kwargs["draft_model_id"] = target.draft_model_id
            if service.unload_model(
                target.model_id,
                **unload_kwargs,
            ):
                unloaded.append(target.model_id)
                if not _visual_surfaces_still_attached(target.model_id):
                    await shutdown_vlm_coordinator(target.model_id)
            release_runtime_surface(
                target.model_id,
                "embeddings",
                adapter_path=target.adapter_path,
                draft_model_id=target.draft_model_id,
            )
        return ModelUnloadResponse(
            task="embeddings",
            status="cleared",
            message=f"Cleared {len(unloaded)} embeddings model(s) from cache",
            unloaded_models=unloaded,
            cache_info=_build_llm_cache_info() if shared_runtime else None,
        )

    if task == "visual":
        from ....embeddings.visual_router import (
            get_loaded_visual_models,
            unload_visual_embedder,
        )
        from ....vision.vlm_batch import shutdown_vlm_coordinator

        unloaded: list[str] = []
        visual_models = sorted(
            set(get_loaded_visual_models()).union(get_attached_models("visual"))
        )
        runtime_targets = get_attached_runtime_targets("visual")
        if not runtime_targets:
            for model_id in visual_models:
                await shutdown_vlm_coordinator(model_id)
                unloaded.extend(
                    unload_visual_embedder(
                        model_id,
                        release_runtime=True,
                    )
                )
            unloaded = list(dict.fromkeys(unloaded))
            return ModelUnloadResponse(
                task="visual",
                status="cleared",
                message=f"Cleared {len(unloaded)} visual model(s) from cache",
                unloaded_models=unloaded,
                cache_info=_build_llm_cache_info(),
            )

        for target in runtime_targets:
            target_remaining = get_remaining_runtime_surfaces(
                target.model_id,
                releasing_surface="visual",
                adapter_path=target.adapter_path,
                draft_model_id=target.draft_model_id,
            )
            preserve_runtime = bool(target_remaining)
            await shutdown_vlm_coordinator(target.model_id)
            unload_kwargs: dict[str, str] = {}
            if target.adapter_path is not None:
                unload_kwargs["adapter_path"] = target.adapter_path
            if target.draft_model_id is not None:
                unload_kwargs["draft_model_id"] = target.draft_model_id
            unloaded.extend(
                unload_visual_embedder(
                    target.model_id,
                    release_runtime=not preserve_runtime,
                    **unload_kwargs,
                )
            )
            release_runtime_surface(
                target.model_id,
                "visual",
                adapter_path=target.adapter_path,
                draft_model_id=target.draft_model_id,
            )
        unloaded = list(dict.fromkeys(unloaded))
        return ModelUnloadResponse(
            task="visual",
            status="cleared",
            message=f"Cleared {len(unloaded)} visual model(s) from cache",
            unloaded_models=unloaded,
            cache_info=_build_llm_cache_info(),
        )

    if task == "stt":
        from ....stt.whisper_model import unload_whisper_model

        unloaded = unload_whisper_model()
        return _build_unload_response(
            task="stt",
            status="cleared",
            message=f"Cleared {len(unloaded)} STT model(s) from cache",
            unloaded_models=unloaded,
        )

    if task == "tts":
        from ....tts.tts_service import TTSService

        unloaded = TTSService.unload_model()
        return _build_unload_response(
            task="tts",
            status="cleared",
            message=f"Cleared {len(unloaded)} TTS model(s) from cache",
            unloaded_models=unloaded,
        )

    if task == "images":
        from ....images.image_runtime import get_image_runtime_pool

        unloaded = await get_image_runtime_pool().clear()
        return _build_unload_response(
            task="images",
            status="cleared",
            message=f"Cleared {len(unloaded)} image model(s) from cache",
            unloaded_models=unloaded,
        )

    raise HTTPException(status_code=400, detail=f"Unsupported task '{task}'")


async def _clear_all_models() -> ModelUnloadResponse:
    from ....batch.coordinator import shutdown_all_coordinators
    from ....vision.vlm_batch import shutdown_all_vlm_coordinators

    await shutdown_all_coordinators()
    await shutdown_all_vlm_coordinators()
    clear_runtime_surface_attachments()
    result = get_models_service().unload_model(model_id=None)
    unloaded_models = list(result.get("unloaded_models", []))

    from ....embeddings.embeddings_service import get_embeddings_service
    from ....embeddings.visual_router import unload_visual_embedder
    from ....images.image_runtime import get_image_runtime_pool
    from ....stt.whisper_model import unload_whisper_model
    from ....tts.tts_service import TTSService

    unloaded_models.extend(_unload_vlm_runtime())
    unloaded_models.extend(get_embeddings_service().clear_models())
    unloaded_models.extend(unload_visual_embedder())
    unloaded_models.extend(await get_image_runtime_pool().clear())
    unloaded_models.extend(unload_whisper_model())
    unloaded_models.extend(TTSService.unload_model())

    result["task"] = None
    result["unloaded_models"] = list(dict.fromkeys(unloaded_models))
    result["message"] = (
        f"Cleared {len(result['unloaded_models'])} model(s) across caches"
    )
    result["cache_info"] = _build_llm_cache_info()
    return ModelUnloadResponse(**result)


@router.post("/models/unload", response_model=ModelUnloadResponse)
@router.post("/v1/models/unload", response_model=ModelUnloadResponse)
async def unload_model(
    request: ModelUnloadRequest | None = None,
    _auth: dict = Depends(verify_auth),
) -> ModelUnloadResponse:
    """
    Unload a model from memory to free VRAM.

    If model ID is provided, unloads that specific model.
    If no model ID is provided, unloads all models from cache.
    """
    try:
        model_id = request.model if request else None
        task_hint = _normalize_task(request.task) if request else None
        adapter_path = request.adapter_path if request else None
        draft_model_id = request.draft_model_id if request else None

        if task_hint and task_hint not in _ALLOWED_TASKS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown task '{task_hint}'. Supported tasks: "
                    + ", ".join(sorted(_ALLOWED_TASKS))
                ),
            )

        if model_id is None and (
            adapter_path is not None or draft_model_id is not None
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "adapter_path and draft_model_id require a specific model "
                    "when unloading runtime variants"
                ),
            )

        if model_id:
            task = _detect_task(model_id, task_hint)
            return await _unload_specific(
                task,
                model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            )

        if task_hint:
            return await _clear_task(task_hint)

        return await _clear_all_models()
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns server status and basic info about loaded models.
    """
    runtime = _snapshot_llm_runtime()
    process_residency = _snapshot_process_residency(runtime)
    cache_info = runtime["cache_info"]

    return {
        "status": "healthy",
        **get_runtime_provenance().health_fields(),
        "loaded_models_count": process_residency["loaded_models_count"],
        "loaded_models": process_residency["loaded_models"],
        "loaded_models_by_backend": process_residency["loaded_models_by_backend"],
        "process_residency": process_residency,
        "loaded_models_runtime": runtime["data"],
        "surface_attachments": runtime["surface_attachments"],
        "runtime_contract": runtime["runtime_contract"],
        "cache_max_size": cache_info.get("max_size", 1),
        "cache_ttl_seconds": cache_info.get("ttl_seconds", 600),
        "memory": _get_runtime_memory_snapshot(),
    }
