from fastapi import APIRouter, HTTPException

from ..chat.mlx.runtime_aliases import resolve_runtime_target
from ..chat.mlx.runtime_policy import endpoint_runtime_session
from .embeddings_service import get_embeddings_service
from .schema import EmbeddingRequest, EmbeddingResponse

router = APIRouter(tags=["embeddings"])
embeddings_service = get_embeddings_service()


@router.post("/embeddings", response_model=EmbeddingResponse)
@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for text input.

    This endpoint generates vector representations of input text,
    which can be used for semantic search, clustering, and other NLP tasks.
    """
    try:
        if embeddings_service.uses_shared_vlm_runtime(request.model):
            target = resolve_runtime_target(request.model)
            if target.adapter_path is None and target.draft_model_id is None:
                canonical_model_id = embeddings_service.canonicalize_model_id(
                    request.model
                )
                async with endpoint_runtime_session(canonical_model_id):
                    return embeddings_service.generate_embeddings(request)
            async with endpoint_runtime_session(
                target.model_id,
                adapter_path=target.adapter_path,
                draft_model_id=target.draft_model_id,
            ):
                return embeddings_service.generate_embeddings(request)
        return embeddings_service.generate_embeddings(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
