from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from .models_service import ModelsService
from .schema import (
    Model,
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
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(repo_id=model_id, filename="config.json"))
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
    if model_type_norm in {"qwen3_vl", "qwen2_vl", "qwen2_5_vl"}:
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
async def list_models(include_details: bool = False) -> ModelList:
    """List all available models"""
    return get_models_service().list_models(include_details)


@router.get("/models/loaded")
@router.get("/v1/models/loaded")
async def list_loaded_models() -> dict:
    """
    List only models currently loaded in memory.

    Unlike /v1/models which lists all cached models on disk,
    this endpoint shows only models actively loaded in runtime caches.
    """
    from ....batch.coordinator import get_loaded_batch_models
    from ....chat.mlx.wrapper_cache import wrapper_cache
    from ....responses.adapter import get_loaded_vlm_models

    loaded_models = wrapper_cache.get_loaded_models()
    vlm_loaded_models = get_loaded_vlm_models()
    batch_loaded_models = get_loaded_batch_models()
    cache_info = wrapper_cache.get_cache_info()
    by_model_id: dict[str, set[str]] = {}

    for model_id in loaded_models:
        by_model_id.setdefault(model_id, set()).add("wrapper")

    for model_id in vlm_loaded_models:
        by_model_id.setdefault(model_id, set()).add("vlm")

    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "loaded": True,
                "task": "llm",
                "backends": sorted(backends),
            }
            for model_id, backends in by_model_id.items()
        ],
        "coordinators": {"llm_batch": batch_loaded_models},
        "caches": {"wrapper": loaded_models, "vlm": vlm_loaded_models},
        "cache_info": cache_info,
    }


@router.get("/models/{model_id:path}", response_model=Model)
@router.get("/v1/models/{model_id:path}", response_model=Model)
async def get_model(model_id: str, include_details: bool = False) -> Model:
    """Get information about a specific model"""
    model = get_models_service().get_model(model_id, include_details)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/models/{model_id:path}", response_model=ModelDeletion)
@router.delete("/v1/models/{model_id:path}", response_model=ModelDeletion)
async def delete_model(request: Request) -> ModelDeletion:
    """
    Delete a fine-tuned model from local cache.
    """
    try:
        model_id = extract_model_id_from_path(request)
        return get_models_service().delete_model(model_id)
    except Exception as e:
        handle_model_error(e)


@router.post("/models/load", response_model=ModelLoadResponse)
@router.post("/v1/models/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    """
    Load a model into memory for inference.

    This endpoint loads an MLX model into the cache, making it ready for
    inference requests. If the model is already loaded, returns success
    with 'already_loaded' status (idempotent).

    The model will be automatically unloaded after the TTL expires (default: 5 min)
    or when cache capacity is reached (LRU eviction).
    """
    try:
        task = _detect_task(request.model, request.task)
        if task == "llm":
            result = get_models_service().load_model(
                model_id=request.model,
                adapter_path=request.adapter_path,
                draft_model_id=request.draft_model_id,
            )
            result["task"] = "llm"
            return ModelLoadResponse(**result)

        if task == "embeddings":
            from ....embeddings.embeddings_service import get_embeddings_service

            service = get_embeddings_service()
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
            from ....embeddings.visual_router import get_visual_embedder

            get_visual_embedder(request.model)
            return ModelLoadResponse(
                id=request.model,
                task="visual",
                status="loaded",
                message=f"Visual embeddings model {request.model} loaded successfully",
                cache_info=None,
            )

        if task == "stt":
            from ....stt.whisper_model import preload_whisper_model

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
            from ....tts.tts_service import TTSService

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

            from ....images.images_service import get_images_service

            service = get_images_service()
            already_loaded = service.load_model(request.model)
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


async def _unload_specific(task: str, model_id: str) -> ModelUnloadResponse:
    if task == "llm":
        from ....batch.coordinator import shutdown_batch_coordinator
        from ....responses.adapter import unload_vlm_model

        await shutdown_batch_coordinator(model_id)
        result = get_models_service().unload_model(model_id=model_id)
        unloaded_models = list(result.get("unloaded_models", []))
        unloaded_models.extend(unload_vlm_model(model_id))
        unloaded_models = list(dict.fromkeys(unloaded_models))

        result["task"] = "llm"
        result["unloaded_models"] = unloaded_models
        if unloaded_models:
            result["status"] = "unloaded"
            result["message"] = f"Model {model_id} unloaded successfully"
        return ModelUnloadResponse(**result)

    if task == "embeddings":
        from ....embeddings.embeddings_service import get_embeddings_service

        service = get_embeddings_service()
        unloaded = [model_id] if service.unload_model(model_id) else []
        status = "unloaded" if unloaded else "not_found"
        message = (
            f"Embeddings model {model_id} unloaded successfully"
            if unloaded
            else f"Embeddings model {model_id} was not loaded"
        )
        return _build_unload_response(
            task="embeddings",
            status=status,
            message=message,
            unloaded_models=unloaded,
        )

    if task == "visual":
        from ....embeddings.visual_router import unload_visual_embedder

        unloaded = unload_visual_embedder(model_id)
        status = "unloaded" if unloaded else "not_found"
        message = (
            f"Visual model {model_id} unloaded successfully"
            if unloaded
            else f"Visual model {model_id} was not loaded"
        )
        return _build_unload_response(
            task="visual",
            status=status,
            message=message,
            unloaded_models=unloaded,
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
        from ....images.images_service import get_images_service

        service = get_images_service()
        unloaded = [model_id] if service.unload_model(model_id) else []
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


async def _clear_task(task: str) -> ModelUnloadResponse:
    if task == "llm":
        from ....batch.coordinator import shutdown_all_coordinators
        from ....responses.adapter import unload_vlm_model

        await shutdown_all_coordinators()
        result = get_models_service().unload_model(model_id=None)
        unloaded_models = list(result.get("unloaded_models", []))
        unloaded_models.extend(unload_vlm_model())
        result["unloaded_models"] = list(dict.fromkeys(unloaded_models))
        result["message"] = (
            f"Cleared {len(result['unloaded_models'])} llm model(s) from cache"
        )
        result["task"] = "llm"
        return ModelUnloadResponse(**result)

    if task == "embeddings":
        from ....embeddings.embeddings_service import get_embeddings_service

        service = get_embeddings_service()
        unloaded = service.clear_models()
        return _build_unload_response(
            task="embeddings",
            status="cleared",
            message=f"Cleared {len(unloaded)} embeddings model(s) from cache",
            unloaded_models=unloaded,
        )

    if task == "visual":
        from ....embeddings.visual_router import unload_visual_embedder

        unloaded = unload_visual_embedder()
        return _build_unload_response(
            task="visual",
            status="cleared",
            message=f"Cleared {len(unloaded)} visual model(s) from cache",
            unloaded_models=unloaded,
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
        from ....images.images_service import get_images_service

        service = get_images_service()
        unloaded = service.clear_models()
        return _build_unload_response(
            task="images",
            status="cleared",
            message=f"Cleared {len(unloaded)} image model(s) from cache",
            unloaded_models=unloaded,
        )

    raise HTTPException(status_code=400, detail=f"Unsupported task '{task}'")


async def _clear_all_models() -> ModelUnloadResponse:
    from ....batch.coordinator import shutdown_all_coordinators

    await shutdown_all_coordinators()
    result = get_models_service().unload_model(model_id=None)
    unloaded_models = list(result.get("unloaded_models", []))

    from ....embeddings.embeddings_service import get_embeddings_service
    from ....embeddings.visual_router import unload_visual_embedder
    from ....images.images_service import get_images_service
    from ....responses.adapter import unload_vlm_model
    from ....stt.whisper_model import unload_whisper_model
    from ....tts.tts_service import TTSService

    unloaded_models.extend(unload_vlm_model())
    unloaded_models.extend(get_embeddings_service().clear_models())
    unloaded_models.extend(unload_visual_embedder())
    unloaded_models.extend(get_images_service().clear_models())
    unloaded_models.extend(unload_whisper_model())
    unloaded_models.extend(TTSService.unload_model())

    result["task"] = None
    result["unloaded_models"] = list(dict.fromkeys(unloaded_models))
    result["message"] = (
        f"Cleared {len(result['unloaded_models'])} model(s) across caches"
    )
    return ModelUnloadResponse(**result)


@router.post("/models/unload", response_model=ModelUnloadResponse)
@router.post("/v1/models/unload", response_model=ModelUnloadResponse)
async def unload_model(
    request: ModelUnloadRequest | None = None,
) -> ModelUnloadResponse:
    """
    Unload a model from memory to free VRAM.

    If model ID is provided, unloads that specific model.
    If no model ID is provided, unloads all models from cache.
    """
    try:
        model_id = request.model if request else None
        task_hint = _normalize_task(request.task) if request else None

        if task_hint and task_hint not in _ALLOWED_TASKS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown task '{task_hint}'. Supported tasks: "
                    + ", ".join(sorted(_ALLOWED_TASKS))
                ),
            )

        if model_id:
            task = _detect_task(model_id, task_hint)
            return await _unload_specific(task, model_id)

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
    from ....chat.mlx.wrapper_cache import wrapper_cache
    from ....responses.adapter import get_loaded_vlm_models

    loaded_models = wrapper_cache.get_loaded_models()
    vlm_loaded_models = get_loaded_vlm_models()
    all_loaded_models = list(dict.fromkeys([*loaded_models, *vlm_loaded_models]))
    cache_info = wrapper_cache.get_cache_info()

    return {
        "status": "healthy",
        "loaded_models_count": len(all_loaded_models),
        "loaded_models": all_loaded_models,
        "loaded_models_by_backend": {
            "wrapper": loaded_models,
            "vlm": vlm_loaded_models,
        },
        "cache_max_size": cache_info.get("max_size", 1),
        "cache_ttl_seconds": cache_info.get("ttl_seconds", 600),
    }
