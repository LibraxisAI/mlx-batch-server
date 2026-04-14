from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from ..chat.mlx.model_types import reset_request_local_runtime_state
from ..chat.mlx.runtime_aliases import resolve_runtime_target
from ..chat.mlx.wrapper_cache import wrapper_cache


@dataclass
class SharedTextEmbeddingResult:
    embeddings: mx.array
    num_tokens: int
    source_type: str


class SharedVLMTextEmbedder:
    """Pool sentence embeddings from a resident shared VLM language tower."""

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> None:
        target = resolve_runtime_target(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        self.model_id = target.model_id
        self.adapter_path = target.adapter_path
        self.draft_model_id = target.draft_model_id
        self.embedding_dim: int | None = None

    def load(self) -> None:
        """Fail fast when the requested model is not a shared VLM runtime."""
        self._get_backend()

    def _get_backend(self) -> tuple[Any, Any]:
        return wrapper_cache.get_vlm_backend(
            self.model_id,
            adapter_path=self.adapter_path,
            draft_model_id=self.draft_model_id,
        )

    @staticmethod
    def _get_child(obj: Any, name: str) -> Any | None:
        if hasattr(obj, name):
            return getattr(obj, name)
        try:
            return obj[name]
        except Exception:
            return None

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if type(value).__module__ == "mlx.core" and type(value).__name__ == "array":
            if value.dtype == mx.bfloat16:
                return np.array(value.astype(mx.float32))
            return np.array(value)
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.array(value)

    @staticmethod
    def _tokenize_text(tokenizer: Any, text: str) -> Any:
        last_error: Exception | None = None
        for tensor_type in ("np", "pt"):
            try:
                return tokenizer(text, return_tensors=tensor_type)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Tokenizer failed for shared VLM text embedding: {last_error}"
        )

    @staticmethod
    def _build_position_ids(batch_size: int, seq_len: int) -> mx.array:
        position_ids = mx.array(np.arange(seq_len, dtype=np.int32)).reshape(1, -1)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))
        return mx.broadcast_to(position_ids[None, ...], (3, batch_size, seq_len))

    def _get_language_model(self, model: Any) -> Any:
        language_model = self._get_child(model, "language_model")
        if language_model is None:
            raise RuntimeError(
                f"Model {self.model_id} does not expose a separable language_model tower"
            )
        return self._get_child(language_model, "model") or language_model

    def _run_language_layers(
        self,
        inner_model: Any,
        inputs_embeds: mx.array,
        position_ids: mx.array,
    ) -> mx.array:
        layers = self._get_child(inner_model, "layers")
        norm = self._get_child(inner_model, "norm")
        if layers is None or norm is None:
            raise RuntimeError(
                f"Model {self.model_id} is missing decoder layers/norm for shared embeddings"
            )

        hidden = inputs_embeds
        for layer in layers:
            try:
                next_hidden = layer(hidden, position_ids=position_ids)
            except TypeError:
                next_hidden = layer(hidden)

            if isinstance(next_hidden, tuple):
                hidden = next_hidden[0]
            elif hasattr(next_hidden, "last_hidden_state"):
                hidden = next_hidden.last_hidden_state
            else:
                hidden = next_hidden

        normalized = norm(hidden)
        if isinstance(normalized, tuple):
            return normalized[0]
        if hasattr(normalized, "last_hidden_state"):
            return normalized.last_hidden_state
        return normalized

    @staticmethod
    def _last_token_pool(
        hidden_states: mx.array,
        attention_mask: mx.array | None,
    ) -> mx.array:
        if attention_mask is None:
            return hidden_states[:, -1, :]

        if bool(mx.all(attention_mask[:, -1]).item()):
            return hidden_states[:, -1, :]

        sequence_lengths = mx.sum(attention_mask, axis=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[mx.arange(batch_size), sequence_lengths]

    @staticmethod
    def _normalize_embeddings(hidden_states: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(hidden_states**2, axis=-1, keepdims=True) + 1e-12)
        return hidden_states / norm

    def embed_text_pooled(self, text: str) -> SharedTextEmbeddingResult:
        with wrapper_cache.vlm_execution(self.model_id):
            model, processor = self._get_backend()
            reset_request_local_runtime_state(model)
            tokenizer = getattr(processor, "tokenizer", processor)
            inputs = self._tokenize_text(tokenizer, text)

            input_ids_np = self._to_numpy(inputs["input_ids"]).astype(
                np.int64, copy=False
            )
            input_ids = mx.array(input_ids_np, dtype=mx.int64)

            attention_mask = inputs.get("attention_mask")
            attention_mask_mx: mx.array | None = None
            if attention_mask is not None:
                attention_mask_np = self._to_numpy(attention_mask).astype(
                    np.int32,
                    copy=False,
                )
                attention_mask_mx = mx.array(attention_mask_np, dtype=mx.int32)

            inner_model = self._get_language_model(model)
            embed_tokens = self._get_child(inner_model, "embed_tokens")
            if embed_tokens is None:
                raise RuntimeError(
                    f"Model {self.model_id} is missing embed_tokens for shared embeddings"
                )

            batch_size, seq_len = input_ids.shape
            inputs_embeds = embed_tokens(input_ids)
            position_ids = self._build_position_ids(batch_size, seq_len)
            hidden_states = self._run_language_layers(
                inner_model,
                inputs_embeds,
                position_ids,
            )

            pooled = self._last_token_pool(hidden_states, attention_mask_mx)
            embeddings = self._normalize_embeddings(pooled).squeeze(0)
            self.embedding_dim = int(embeddings.shape[-1])
            mx.eval(embeddings)

            num_tokens = (
                int(mx.sum(attention_mask_mx).item())
                if attention_mask_mx is not None
                else int(input_ids.shape[1])
            )

            del hidden_states
            del pooled
            del attention_mask_mx
            mx.clear_cache()

            return SharedTextEmbeddingResult(
                embeddings=embeddings,
                num_tokens=num_tokens,
                source_type="text",
            )
