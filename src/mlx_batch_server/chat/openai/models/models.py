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


def get_models_service() -> ModelsService:
    """Get or create the models service singleton with lazy initialization."""
    global _models_service
    if _models_service is None:
        _models_service = ModelsService()
    return _models_service


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
        result = get_models_service().load_model(
            model_id=request.model,
            adapter_path=request.adapter_path,
            draft_model_id=request.draft_model_id,
        )
        return ModelLoadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        result = get_models_service().unload_model(model_id=model_id)
        return ModelUnloadResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
