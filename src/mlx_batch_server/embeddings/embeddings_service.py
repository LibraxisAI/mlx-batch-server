import os
from typing import Any

import mlx.core as mx
import numpy as np
import tiktoken
from mlx_embeddings import generate, load

from ..utils.logger import logger
from .schema import EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage


class EmbeddingsService:
    """Service for generating embeddings using MLX models (focused on BERT-like models)"""

    def __init__(self):
        # Map of loaded models for caching
        self._models: dict[str, tuple[Any, Any]] = {}
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

    def _get_model(self, model_id: str) -> tuple[Any, Any]:
        """Get or load a model based on its ID"""
        if model_id not in self._models:
            logger.info(f"Loading embedding model: {model_id}")
            if self._should_use_qwen3_vl_fallback(model_id):
                logger.warning(
                    "mlx-embeddings missing qwen3_vl support. "
                    "Using text-only qwen3 fallback."
                )
                try:
                    model, processor = self._load_qwen3_vl_text_model(model_id)
                    self._models[model_id] = (model, processor)
                    return self._models[model_id]
                except Exception as fallback_error:
                    logger.error(
                        f"Error loading qwen3_vl text fallback for {model_id}: "
                        f"{fallback_error!s}"
                    )
                    raise RuntimeError(
                        f"Failed to load qwen3_vl embedding model: {fallback_error!s}"
                    ) from fallback_error
            try:
                model, processor = load(model_id)
                self._models[model_id] = (model, processor)
            except Exception as e:
                err = str(e)
                if "qwen3_vl" in err.lower():
                    logger.warning(
                        "Model type qwen3_vl not supported by mlx-embeddings. "
                        "Falling back to text-only qwen3 loader."
                    )
                    try:
                        model, processor = self._load_qwen3_vl_text_model(model_id)
                        self._models[model_id] = (model, processor)
                    except Exception as fallback_error:
                        logger.error(
                            f"Error loading qwen3_vl text fallback for {model_id}: "
                            f"{fallback_error!s}"
                        )
                        raise RuntimeError(
                            f"Failed to load qwen3_vl embedding model: {fallback_error!s}"
                        ) from fallback_error
                else:
                    logger.error(f"Error loading embedding model {model_id}: {e!s}")
                    raise RuntimeError(f"Failed to load embedding model: {e!s}") from e

        return self._models[model_id]

    def load_model(self, model_id: str) -> bool:
        """Preload an embeddings model. Returns True if it was already loaded."""
        already_loaded = model_id in self._models
        self._get_model(model_id)
        return already_loaded

    def unload_model(self, model_id: str) -> bool:
        """Unload a specific embeddings model. Returns True if it was loaded."""
        if model_id in self._models:
            self._models.pop(model_id, None)
            mx.clear_cache()
            return True
        return False

    def clear_models(self) -> list[str]:
        """Unload all embeddings models and return the unloaded model IDs."""
        unloaded = list(self._models.keys())
        if unloaded:
            self._models.clear()
            mx.clear_cache()
        return unloaded

    def _should_use_qwen3_vl_fallback(self, model_id: str) -> bool:
        """Return True when qwen3_vl is requested but not supported by mlx-embeddings."""
        model_lower = model_id.lower()
        if "qwen3-vl" not in model_lower and "qwen3_vl" not in model_lower:
            return False

        if os.environ.get("MLX_EMBED_USE_MLX_EMBEDDINGS_QWEN3_VL") == "1":
            return False

        try:
            import importlib.util

            if importlib.util.find_spec("mlx_embeddings.models.qwen3_vl") is not None:
                return False
        except Exception:
            return True

        return True

    def _load_qwen3_vl_text_model(self, model_id: str) -> tuple[Any, Any]:
        """Load Qwen3-VL embeddings using the text tower only (mlx-lm weights)."""
        from pathlib import Path

        from mlx import nn
        from mlx_embeddings.models import qwen3 as qwen3_embeddings
        from mlx_embeddings.tokenizer_utils import load_tokenizer
        from mlx_embeddings.utils import get_model_path, load_config

        model_path = Path(get_model_path(model_id))
        config = load_config(model_path)
        text_config = config.get("text_config") or {}

        model_args = qwen3_embeddings.ModelArgs.from_dict(text_config)
        model = qwen3_embeddings.Model(model_args)

        weight_files = list(model_path.rglob("model*.safetensors"))
        if not weight_files:
            weight_files = list(model_path.glob("weight*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        weights = {}
        for wf in weight_files:
            loaded_weights = mx.load(str(wf))
            if wf.parent != model_path:
                folder_name = wf.parent.name
                loaded_weights = {
                    f"{folder_name}.{k}": v for k, v in loaded_weights.items()
                }
            for raw_key, value in loaded_weights.items():
                if not raw_key.startswith("language_model."):
                    continue
                clean_key = raw_key[len("language_model.") :]
                if "lm_head." in clean_key:
                    continue
                weights[clean_key] = value

        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        quantization = config.get("quantization") or config.get("quantization_config")
        if quantization:

            def class_predicate(p, m):
                if not hasattr(m, "to_quantized"):
                    return False
                return f"{p}.scales" in weights

            nn.quantize(model, **quantization, class_predicate=class_predicate)

        model.load_weights(list(weights.items()))
        tokenizer = load_tokenizer(model_path)
        return model, tokenizer

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
        model, processor = self._get_model(model_id)

        # Handle both string and list of strings
        inputs = request.input if isinstance(request.input, list) else [request.input]

        # Count tokens for usage info
        token_count = self._count_tokens(inputs)

        # Generate embeddings for all inputs
        embeddings = []
        for idx, text in enumerate(inputs):
            try:
                # Generate embedding using the model
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
                    embedding = self._extract_output_embeddings(output, model_id)

                # Convert to list of floats with proper formatting
                embedding_list = self._ensure_float_list(embedding)

                # Create embedding data
                embedding_data = EmbeddingData(
                    embedding=embedding_list,
                    index=idx,
                )
                embeddings.append(embedding_data)

            except Exception as e:
                logger.error(f"Error generating embedding: {e!s}", exc_info=True)
                raise RuntimeError(f"Failed to generate embedding: {e!s}") from e

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
