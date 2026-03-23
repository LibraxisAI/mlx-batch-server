from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .qwen3_vl_embedder import Qwen3VLEmbedder

router = APIRouter(tags=["visual-embeddings"])

_embedder_cache: dict[tuple[str, str | None, str | None], Qwen3VLEmbedder] = {}
_embedder_lock = threading.Lock()


class VisualEmbeddingRequest(BaseModel):
    model: str = Field(..., description="Qwen3-VL model ID or local path")
    images: list[str] | None = Field(default=None, description="Base64 or file paths")
    texts: list[str] | None = Field(default=None, description="Text queries")
    pdf_path: str | None = Field(default=None, description="Local PDF path")
    max_pages: int | None = Field(default=None, description="Limit PDF pages")
    projection_path: str | None = Field(default=None, description="Projection weights")
    processor_id: str | None = Field(default=None, description="HF processor ID")


class MaxSimRequest(BaseModel):
    query_embedding: list[list[float]]
    doc_embedding: list[list[float]]


def _get_embedder(
    model_id: str, projection_path: str | None, processor_id: str | None
) -> Qwen3VLEmbedder:
    key = (model_id, projection_path, processor_id)
    with _embedder_lock:
        embedder = _embedder_cache.get(key)
        if embedder is None:
            embedder = Qwen3VLEmbedder(
                model_id=model_id,
                projection_path=projection_path,
                processor_id=processor_id,
            )
            embedder.load()
            embedder.log_summary()
            _embedder_cache[key] = embedder
    return embedder


def get_visual_embedder(
    model_id: str, projection_path: str | None = None, processor_id: str | None = None
) -> Qwen3VLEmbedder:
    """Return a shared visual embedder instance for the given model."""
    return _get_embedder(model_id, projection_path, processor_id)


def unload_visual_embedder(model_id: str | None = None) -> list[str]:
    """Unload visual embedders. Returns unloaded model IDs."""
    with _embedder_lock:
        if model_id is None:
            unloaded = list({key[0] for key in _embedder_cache})
            _embedder_cache.clear()
            return unloaded

        keys_to_remove = [key for key in _embedder_cache if key[0] == model_id]
        for key in keys_to_remove:
            _embedder_cache.pop(key, None)
        return [model_id] if keys_to_remove else []


@router.post("/visual-embeddings")
@router.post("/v1/visual-embeddings")
async def create_visual_embeddings(request: VisualEmbeddingRequest) -> dict[str, Any]:
    if not any([request.images, request.texts, request.pdf_path]):
        raise HTTPException(
            status_code=400, detail="Provide images, texts, or pdf_path."
        )

    try:
        embedder = _get_embedder(
            request.model, request.projection_path, request.processor_id
        )
        response: dict[str, Any] = {
            "object": "embedding_response",
            "model": request.model,
            "dim": embedder.embedding_dim,
        }

        if request.texts:
            text_embeddings = []
            for text in request.texts:
                result = embedder.embed_text(text)
                text_embeddings.append(
                    {
                        "embedding": embedder.to_numpy(result).tolist(),
                        "num_tokens": result.num_tokens,
                        "source_type": "text",
                    }
                )
            response["text_embeddings"] = text_embeddings

        if request.images:
            image_embeddings = []
            for img in request.images:
                result = embedder.embed_image(img)
                image_embeddings.append(
                    {
                        "embedding": embedder.to_numpy(result).tolist(),
                        "num_tokens": result.num_tokens,
                        "source_type": "image",
                    }
                )
            response["image_embeddings"] = image_embeddings

        if request.pdf_path:
            pdf_embeddings = embedder.embed_pdf(
                request.pdf_path, max_pages=request.max_pages
            )
            response["pdf_embeddings"] = [
                {
                    "page": i,
                    "embedding": embedder.to_numpy(result).tolist(),
                    "num_tokens": result.num_tokens,
                    "source_type": result.source_type,
                }
                for i, result in enumerate(pdf_embeddings)
            ]

        if response["dim"] is None:
            sample = (
                (response.get("text_embeddings") or response.get("image_embeddings"))
                or response.get("pdf_embeddings")
                or [{}]
            )[0]
            if isinstance(sample.get("embedding"), list):
                response["dim"] = (
                    len(sample["embedding"][0]) if sample["embedding"] else None
                )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/maxsim")
@router.post("/v1/maxsim")
async def compute_maxsim(request: MaxSimRequest) -> dict[str, float]:
    try:
        score = Qwen3VLEmbedder.maxsim_score(
            request.query_embedding, request.doc_embedding
        )
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
