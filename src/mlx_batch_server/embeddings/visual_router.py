from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.dependency import verify_auth
from ..chat.mlx.runtime_aliases import (
    normalize_runtime_model_id,
    normalize_runtime_path,
    resolve_runtime_target,
)
from ..chat.mlx.runtime_attachments import (
    attach_runtime_surface,
    release_runtime_surface,
)
from ..chat.mlx.runtime_policy import endpoint_runtime_session
from ..chat.mlx.wrapper_cache import wrapper_cache
from .qwen3_vl_embedder import Qwen3VLEmbedder

router = APIRouter(tags=["visual-embeddings"])

# Keep the unload seam monkeypatchable while delegating real residency
# ownership to the unified wrapper cache.
unload_vlm_model = wrapper_cache.unload_vlm_model

_embedder_cache: dict[
    tuple[str, str | None, str | None, str | None, str | None],
    Qwen3VLEmbedder,
] = {}
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


def _normalize_embedder_key(
    model_id: str,
    projection_path: str | None,
    processor_id: str | None,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    return (
        target.model_id,
        target.adapter_path,
        target.draft_model_id,
        normalize_runtime_path(projection_path),
        normalize_runtime_model_id(processor_id) if processor_id else None,
    )


def _get_embedder(
    model_id: str,
    projection_path: str | None,
    processor_id: str | None,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> Qwen3VLEmbedder:
    key = _normalize_embedder_key(
        model_id,
        projection_path,
        processor_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _embedder_lock:
        embedder = _embedder_cache.get(key)
        if embedder is None:
            embedder = Qwen3VLEmbedder(
                model_id=key[0],
                adapter_path=key[1],
                draft_model_id=key[2],
                projection_path=key[3],
                processor_id=key[4],
            )
            embedder.load()
            embedder.log_summary()
            _embedder_cache[key] = embedder
    attach_runtime_surface(
        key[0],
        "visual",
        adapter_path=key[1],
        draft_model_id=key[2],
    )
    return embedder


def get_visual_embedder(
    model_id: str,
    projection_path: str | None = None,
    processor_id: str | None = None,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> Qwen3VLEmbedder:
    """Return a shared visual embedder instance for the given model."""
    return _get_embedder(
        model_id,
        projection_path,
        processor_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )


def get_loaded_visual_models() -> list[str]:
    """Return canonical model ids currently attached to visual embedders."""
    with _embedder_lock:
        return sorted({key[0] for key in _embedder_cache})


def has_visual_embedder(
    model_id: str,
    *,
    projection_path: str | None = None,
    processor_id: str | None = None,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> bool:
    """Return True when the exact visual embedder runtime key is cached."""
    key = _normalize_embedder_key(
        model_id,
        projection_path,
        processor_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with _embedder_lock:
        return key in _embedder_cache


def unload_visual_embedder(
    model_id: str | None = None,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
    release_runtime: bool = True,
) -> list[str]:
    """Unload visual embedders and their shared VLM runtime residency."""
    removed_models: list[str] = []
    runtime_targets: list[tuple[str, str | None, str | None]] = []
    specific_model_id = (
        normalize_runtime_model_id(model_id) if model_id is not None else None
    )
    specific_target = (
        resolve_runtime_target(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        if model_id is not None
        else None
    )

    with _embedder_lock:
        if model_id is None:
            removed_models = sorted({key[0] for key in _embedder_cache})
            runtime_targets = sorted(
                {(key[0], key[1], key[2]) for key in _embedder_cache},
            )
            _embedder_cache.clear()
        else:
            assert specific_model_id is not None
            assert specific_target is not None
            if adapter_path is None and draft_model_id is None:
                keys_to_remove = [
                    key for key in _embedder_cache if key[0] == specific_model_id
                ]
            else:
                keys_to_remove = [
                    key
                    for key in _embedder_cache
                    if key[:3]
                    == (
                        specific_target.model_id,
                        specific_target.adapter_path,
                        specific_target.draft_model_id,
                    )
                ]
            for key in keys_to_remove:
                _embedder_cache.pop(key, None)
            runtime_targets = sorted(
                {(key[0], key[1], key[2]) for key in keys_to_remove}
            )
            removed_models = [specific_model_id] if keys_to_remove else []

    if specific_target is not None and not runtime_targets:
        runtime_targets = [
            (
                specific_target.model_id,
                specific_target.adapter_path,
                specific_target.draft_model_id,
            )
        ]

    runtime_unloaded: list[str] = []
    for (
        runtime_model_id,
        runtime_adapter_path,
        runtime_draft_model_id,
    ) in runtime_targets:
        attachment_state = release_runtime_surface(
            runtime_model_id,
            "visual",
            adapter_path=runtime_adapter_path,
            draft_model_id=runtime_draft_model_id,
        )
        if not release_runtime or attachment_state.remaining_surfaces:
            continue
        unload_kwargs: dict[str, str] = {}
        if runtime_adapter_path is not None:
            unload_kwargs["adapter_path"] = runtime_adapter_path
        if runtime_draft_model_id is not None:
            unload_kwargs["draft_model_id"] = runtime_draft_model_id
        runtime_unloaded.extend(
            unload_vlm_model(
                runtime_model_id,
                **unload_kwargs,
            )
        )

    return list(dict.fromkeys([*removed_models, *runtime_unloaded]))


@router.post("/visual-embeddings")
@router.post("/v1/visual-embeddings")
async def create_visual_embeddings(
    request: VisualEmbeddingRequest,
    _auth: dict = Depends(verify_auth),
) -> dict[str, Any]:
    if not any([request.images, request.texts, request.pdf_path]):
        raise HTTPException(
            status_code=400, detail="Provide images, texts, or pdf_path."
        )

    try:
        target = resolve_runtime_target(request.model)
        async with endpoint_runtime_session(
            target.model_id,
            adapter_path=target.adapter_path,
            draft_model_id=target.draft_model_id,
        ):
            embedder = _get_embedder(
                request.model,
                request.projection_path,
                request.processor_id,
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
                    (
                        response.get("text_embeddings")
                        or response.get("image_embeddings")
                    )
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
async def compute_maxsim(
    request: MaxSimRequest,
    _auth: dict = Depends(verify_auth),
) -> dict[str, float]:
    try:
        score = Qwen3VLEmbedder.maxsim_score(
            request.query_embedding, request.doc_embedding
        )
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
