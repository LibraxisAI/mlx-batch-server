from typing import Any

import mlx.core as mx
import numpy as np
import tiktoken
from mlx_embeddings import generate, load

from ..chat.mlx.model_types import resolves_to_multimodal_runtime
from ..chat.mlx.runtime_aliases import resolve_runtime_model_id
from ..chat.mlx.runtime_attachments import (
    attach_runtime_surface,
    get_attached_models,
    release_runtime_surface,
)
from ..chat.mlx.wrapper_cache import normalize_model_id, wrapper_cache
from ..utils.logger import logger
from .schema import EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage
from .shared_vlm_text_embedder import SharedVLMTextEmbedder


class EmbeddingsService:
    """Service for generating embeddings using MLX models (focused on BERT-like models)"""

    def __init__(self):
        # Map of loaded models for caching
        self._models: dict[str, tuple[Any, Any]] = {}
        # Qwen3-VL embeddings reuse the shared visual/VLM runtime instead of
        # loading a second text-only fallback into a private cache.
        self._shared_vlm_models: set[str] = set()
        # Default encoder for token counting
        try:
            self._default_tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to another common encoding if cl100k_base is not available
            try:
                self._default_tokenizer = tiktoken.get_encoding("p50k_base")
            except Exception:
                logger.warning(
                    "Could not load any tiktoken encoding, token counts may be inaccurate"
                )
                self._default_tokenizer = None

    def canonicalize_model_id(self, model_id: str) -> str:
        """Resolve aliases and normalize remote IDs for stable cache keys."""
        return normalize_model_id(resolve_runtime_model_id(model_id))

    def _get_model(self, model_id: str) -> tuple[Any, Any]:
        """Get or load a model based on its ID"""
        canonical_model_id = self.canonicalize_model_id(model_id)
        if canonical_model_id not in self._models:
            logger.info(f"Loading embedding model: {canonical_model_id}")
            try:
                model, processor = load(canonical_model_id)
                self._models[canonical_model_id] = (model, processor)
            except Exception as e:
                logger.error(
                    f"Error loading embedding model {canonical_model_id}: {e!s}"
                )
                raise RuntimeError(f"Failed to load embedding model: {e!s}") from e

        return self._models[canonical_model_id]

    def load_model(self, model_id: str) -> bool:
        """Preload an embeddings model. Returns True if it was already loaded."""
        if self._should_use_shared_vlm_embeddings(model_id):
            canonical_model_id = self.canonicalize_model_id(model_id)
            already_loaded = canonical_model_id in self._shared_vlm_models
            if not already_loaded:
                self._get_shared_vlm_embedder(canonical_model_id)
                self._shared_vlm_models.add(canonical_model_id)
            return already_loaded

        canonical_model_id = self.canonicalize_model_id(model_id)
        already_loaded = canonical_model_id in self._models
        self._get_model(canonical_model_id)
        return already_loaded

    def get_loaded_native_models(self) -> list[str]:
        """Return non-shared embeddings model ids currently cached privately."""
        return list(self._models.keys())

    def clear_native_models(self) -> list[str]:
        """Clear only the private embeddings cache, leaving shared VLM runtime alone."""
        unloaded = list(self._models.keys())
        if unloaded:
            self._models.clear()
            mx.clear_cache()
        return unloaded

    def get_shared_vlm_models(self) -> list[str]:
        """Return canonical shared-runtime VLM model ids seen by the service."""
        return sorted(self._shared_vlm_models)

    def unload_model(self, model_id: str, *, release_runtime: bool = True) -> bool:
        """Unload a specific embeddings model. Returns True if it was loaded."""
        if self._should_use_shared_vlm_embeddings(model_id):
            canonical_model_id = self.canonicalize_model_id(model_id)
            was_loaded = canonical_model_id in self._shared_vlm_models
            self._shared_vlm_models.discard(canonical_model_id)
            attachment_state = release_runtime_surface(
                canonical_model_id,
                "embeddings",
            )
            if not release_runtime or attachment_state.remaining_surfaces:
                return was_loaded or attachment_state.was_attached
            unloaded = self._unload_shared_vlm_embedder(canonical_model_id)
            return was_loaded or bool(unloaded)

        canonical_model_id = self.canonicalize_model_id(model_id)
        if canonical_model_id in self._models:
            self._models.pop(canonical_model_id, None)
            mx.clear_cache()
            return True
        return False

    def clear_models(self, *, release_runtime: bool = True) -> list[str]:
        """Unload all embeddings models and return the unloaded model IDs."""
        unloaded = self.clear_native_models()
        shared_vlm_models = sorted(
            set(self.get_shared_vlm_models()).union(get_attached_models("embeddings"))
        )
        self._shared_vlm_models.clear()
        for model_id in shared_vlm_models:
            attachment_state = release_runtime_surface(model_id, "embeddings")
            if release_runtime and not attachment_state.remaining_surfaces:
                self._unload_shared_vlm_embedder(model_id)
        return list(dict.fromkeys([*unloaded, *shared_vlm_models]))

    def _should_use_shared_vlm_embeddings(self, model_id: str) -> bool:
        """Route any multimodal model through the shared VLM runtime spine."""
        return resolves_to_multimodal_runtime(model_id)

    def uses_shared_vlm_runtime(self, model_id: str) -> bool:
        """Expose whether this embeddings request rides on the shared VLM cache."""
        return self._should_use_shared_vlm_embeddings(model_id)

    def has_shared_vlm_runtime_models(self) -> bool:
        """Return True when shared VLM embeddings models are currently attached."""
        return bool(self._shared_vlm_models)

    def _get_shared_vlm_embedder(self, model_id: str) -> SharedVLMTextEmbedder:
        """Pool text embeddings directly from the shared resident VLM runtime."""
        embedder = SharedVLMTextEmbedder(model_id)
        embedder.load()
        attach_runtime_surface(model_id, "embeddings")
        return embedder

    def _unload_shared_vlm_embedder(self, model_id: str) -> list[str]:
        """Release the shared VLM runtime backing multimodal text embeddings."""
        return wrapper_cache.unload_vlm_model(model_id)

    def _count_tokens(self, text: str | list[str]) -> int:
        """Count tokens in input text"""
        if self._default_tokenizer is None:
            # If no tokenizer is available, use a simple approximation
            if isinstance(text, str):
                return len(text.split())
            elif isinstance(text, list):
                return sum(len(t.split()) for t in text)
            return 0

        try:
            if isinstance(text, str):
                return len(self._default_tokenizer.encode(text))
            elif isinstance(text, list):
                return sum(len(self._default_tokenizer.encode(t)) for t in text)
        except Exception as e:
            logger.warning(f"Error counting tokens: {e!s}. Using fallback method.")
            # Fallback to simple approximation
            if isinstance(text, str):
                return len(text.split())
            elif isinstance(text, list):
                return sum(len(t.split()) for t in text)
        return 0

    def _ensure_float_list(self, embedding) -> list[float]:
        """Ensure embedding is a flat list of float values"""
        if isinstance(embedding, list):
            # Handle case where first element is itself a list or array
            if len(embedding) > 0 and isinstance(
                embedding[0], (list, mx.array, np.ndarray)
            ):
                return [float(x) for x in embedding[0]]
            # Otherwise, convert each element to float
            return [float(x) for x in embedding]
        if isinstance(embedding, (mx.array, np.ndarray)):
            # Ensure array is 1D
            if embedding.ndim > 1:
                embedding = embedding.reshape(-1)
            return [float(x) for x in embedding.tolist()]
        # Handle any other unexpected type
        return [float(x) for x in list(embedding)]

    def _get_bert_embeddings(self, model, processor, text, model_id):
        """Extract embeddings specifically for BERT-like models"""
        # Use proper encode method - processor may have different method names
        encode_method = getattr(processor, "encode", None)
        if encode_method is None:
            encode_method = getattr(processor, "batch_encode_plus", None)

        if encode_method:
            # Use encode or batch_encode_plus
            input_ids = encode_method(text, return_tensors="mlx")

            # Handle different input formats
            if isinstance(input_ids, dict):
                outputs = model(**input_ids)
            else:
                outputs = model(input_ids)

            return self._extract_output_embeddings(outputs, model_id)

        raise ValueError("Could not determine how to extract embeddings from model")

    def _extract_output_embeddings(self, output, model_id: str | None):
        """Extract embeddings from common MLX model output types."""
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output

        if (
            hasattr(output, "last_hidden_state")
            and output.last_hidden_state is not None
        ):
            model_name = (model_id or "").lower()
            if "minilm" in model_name:
                return output.last_hidden_state[:, 0, :]
            return output.last_hidden_state.mean(axis=1)

        return output

    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings based on the request"""
        model_id = request.model
        # Handle both string and list of strings
        inputs = request.input if isinstance(request.input, list) else [request.input]

        # Count tokens for usage info
        token_count = self._count_tokens(inputs)
        shared_vlm_embedder = None
        model = None
        processor = None
        if self._should_use_shared_vlm_embeddings(model_id):
            canonical_model_id = self.canonicalize_model_id(model_id)
            shared_vlm_embedder = self._get_shared_vlm_embedder(canonical_model_id)
            self._shared_vlm_models.add(canonical_model_id)
            token_count = 0
        else:
            model, processor = self._get_model(model_id)

        # Generate embeddings for all inputs
        embeddings = []
        try:
            for idx, text in enumerate(inputs):
                embedding = None
                try:
                    if shared_vlm_embedder is not None:
                        result = shared_vlm_embedder.embed_text_pooled(text)
                        embedding_list = self._ensure_float_list(result.embeddings)
                        token_count += result.num_tokens
                    else:
                        try:
                            # First try the specific BERT extraction method
                            embedding = self._get_bert_embeddings(
                                model, processor, text, model_id
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed with BERT method: {e!s}. "
                                "Trying general generate() function."
                            )
                            # Fall back to the generate function
                            output = generate(model, processor, text)
                            embedding = self._extract_output_embeddings(
                                output, model_id
                            )

                        # Finalize lazy MLX graphs before converting to Python
                        # floats so long embedding runs do not pin intermediates.
                        if isinstance(embedding, mx.array):
                            mx.eval(embedding)

                        # Convert to list of floats with proper formatting
                        embedding_list = self._ensure_float_list(embedding)
                        embedding = None

                    # Create embedding data
                    embedding_data = EmbeddingData(
                        embedding=embedding_list,
                        index=idx,
                    )
                    embeddings.append(embedding_data)

                except Exception as e:
                    logger.error(f"Error generating embedding: {e!s}", exc_info=True)
                    raise RuntimeError(f"Failed to generate embedding: {e!s}") from e
        finally:
            mx.clear_cache()

        # Create the full response
        response = EmbeddingResponse(
            data=embeddings,
            model=model_id,
            usage=EmbeddingUsage(prompt_tokens=token_count, total_tokens=token_count),
        )

        return response


_embeddings_service: EmbeddingsService | None = None


def get_embeddings_service() -> EmbeddingsService:
    """Return a shared embeddings service instance."""
    global _embeddings_service
    if _embeddings_service is None:
        _embeddings_service = EmbeddingsService()
    return _embeddings_service
