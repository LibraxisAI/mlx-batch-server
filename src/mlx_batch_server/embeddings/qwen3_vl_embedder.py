from __future__ import annotations

import base64
import io
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor

from ..chat.mlx.model_types import reset_request_local_runtime_state
from ..chat.mlx.runtime_aliases import (
    normalize_runtime_model_id,
    normalize_runtime_path,
    resolve_runtime_target,
)
from ..utils.logger import logger
from ..vision.vlm_cache import get_vlm_backend, vlm_execution

try:
    from mlx_vlm.utils import get_model_path
except Exception as exc:
    get_model_path = None
    _mlx_vlm_import_error = exc
else:
    _mlx_vlm_import_error = None

try:
    import fitz
except Exception as exc:
    fitz = None
    _fitz_import_error = exc
else:
    _fitz_import_error = None

IMAGE_PAD_TOKEN = 151655


@dataclass
class EmbeddingResult:
    embeddings: mx.array
    num_tokens: int
    source_type: str


class Qwen3VLEmbedder:
    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
        projection_path: str | None = None,
        processor_id: str | None = None,
    ) -> None:
        target = resolve_runtime_target(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        self.model_id = target.model_id
        self.adapter_path = target.adapter_path
        self.draft_model_id = target.draft_model_id
        self.projection_path = normalize_runtime_path(
            projection_path or os.environ.get("QWEN3_VL_PROJECTION_PATH")
        )
        raw_processor_id = processor_id or os.environ.get("QWEN3_VL_PROCESSOR_ID")
        self.processor_id = (
            normalize_runtime_model_id(raw_processor_id) if raw_processor_id else None
        )

        self.model: Any | None = None
        self.processor: Any | None = None
        self.tomoro_processor: Any | None = None
        self.embedding_dim: int | None = None
        self._proj_weight: mx.array | None = None
        self._proj_bias: mx.array | None = None
        self._image_token_id = IMAGE_PAD_TOKEN
        self._loaded = False
        self._has_vision = False

    def _get_backend(self) -> tuple[Any, Any]:
        """Resolve the shared resident VLM backend from the unified runtime cache."""
        return get_vlm_backend(
            self.model_id,
            adapter_path=self.adapter_path,
            draft_model_id=self.draft_model_id,
            surface="visual",
        )

    def load(self) -> None:
        if self._loaded:
            return

        if get_model_path is None:
            raise RuntimeError(
                "mlx-vlm is required for qwen3_vl visual embeddings."
            ) from _mlx_vlm_import_error

        model_path = get_model_path(self.model_id)
        self._has_vision = self._detect_vision_assets(model_path)

        model, processor = self._get_backend()
        if model is None or processor is None:
            raise RuntimeError("mlx-vlm returned an empty model/processor")

        processor_id = self.processor_id or self.model_id
        self.tomoro_processor = AutoProcessor.from_pretrained(
            processor_id, trust_remote_code=True
        )

        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            for source in self._processor_sources(processor):
                if source is None:
                    continue
                image_token_id = getattr(source, "image_token_id", None) or getattr(
                    source, "image_token_index", None
                )
                if image_token_id is not None:
                    break
        self._image_token_id = image_token_id or IMAGE_PAD_TOKEN

        if self.projection_path:
            self._load_projection(self.projection_path)

        self._loaded = True

    def _detect_vision_assets(self, model_path: Path) -> bool:
        preprocessor = model_path / "preprocessor_config.json"
        if preprocessor.exists():
            return True

        index_path = model_path / "model.safetensors.index.json"
        if not index_path.exists():
            return False

        try:
            data = index_path.read_text(encoding="utf-8")
        except OSError:
            return False

        return "vision_tower" in data or "vision_model" in data

    def _load_projection(self, projection_path: str) -> None:
        path = Path(projection_path)
        if not path.exists():
            raise FileNotFoundError(f"Projection weights not found: {path}")

        with safe_open(str(path), framework="numpy") as f:
            keys = set(f.keys())
            if "embedding_proj_layer.weight" in keys:
                weight = f.get_tensor("embedding_proj_layer.weight")
                bias = f.get_tensor("embedding_proj_layer.bias")
            elif "weight" in keys:
                weight = f.get_tensor("weight")
                bias = f.get_tensor("bias")
            else:
                raise ValueError("Projection weights missing expected keys")

        self._proj_weight = mx.array(weight)
        self._proj_bias = mx.array(bias)
        self.embedding_dim = int(weight.shape[0])

    def _project_and_normalize(self, hidden_states: mx.array) -> mx.array:
        if self._proj_weight is not None and self._proj_bias is not None:
            embeddings = hidden_states @ self._proj_weight.T + self._proj_bias
        else:
            embeddings = hidden_states
            if self.embedding_dim is None:
                self.embedding_dim = int(embeddings.shape[-1])

        norm = mx.sqrt(mx.sum(embeddings**2, axis=-1, keepdims=True) + 1e-12)
        return embeddings / norm

    def _build_position_ids(self, batch_size: int, seq_len: int) -> mx.array:
        position_ids = mx.array(np.arange(seq_len, dtype=np.int32)).reshape(1, -1)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))
        return mx.broadcast_to(position_ids[None, ...], (3, batch_size, seq_len))

    @staticmethod
    def _to_numpy(value):
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
    def _get_child(obj, name):
        if hasattr(obj, name):
            return getattr(obj, name)
        try:
            return obj[name]
        except Exception:
            return None

    def _get_language_model(self, model: Any):
        language_model = self._get_child(model, "language_model")
        if language_model is None:
            raise RuntimeError("language_model not found in qwen3_vl model")
        inner_model = self._get_child(language_model, "model") or language_model
        return inner_model

    def _get_vision_tower(self, model: Any):
        vision_tower = self._get_child(model, "vision_tower")
        if vision_tower is None:
            raise RuntimeError("vision_tower not found in qwen3_vl model")
        return vision_tower

    def _run_language_layers(
        self, inner_model, inputs_embeds: mx.array, position_ids: mx.array
    ) -> mx.array:
        layers = self._get_child(inner_model, "layers")
        norm = self._get_child(inner_model, "norm")
        if layers is None or norm is None:
            raise RuntimeError("language model layers/norm missing for qwen3_vl")

        h = inputs_embeds
        for layer in layers:
            h = layer(h, position_ids=position_ids)
        return norm(h)

    def _count_text_tokens(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None,
    ) -> int:
        if attention_mask is None:
            return int(input_ids.shape[1])
        return int(mx.sum(attention_mask).item())

    def _last_token_pool(
        self,
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

    def _processor_call(self, **kwargs):
        last_error = None
        for tensor_type in ("np", "pt"):
            try:
                return self.tomoro_processor(return_tensors=tensor_type, **kwargs)
            except Exception as e:
                last_error = e
        raise RuntimeError(f"Processor failed for inputs: {last_error}") from last_error

    def _tokenize_text(self, tokenizer, text: str):
        last_error = None
        for tensor_type in ("np", "pt"):
            try:
                return tokenizer(text, return_tensors=tensor_type)
            except Exception as e:
                last_error = e
        raise RuntimeError(f"Tokenizer failed for text: {last_error}") from last_error

    def _processor_sources(self, processor: Any):
        return (
            self.tomoro_processor,
            getattr(self.tomoro_processor, "tokenizer", None),
            processor,
            getattr(processor, "tokenizer", None),
        )

    def _find_image_token_text(self, processor: Any) -> str | None:
        for source in self._processor_sources(processor):
            if source is None:
                continue
            image_token = getattr(source, "image_token", None)
            if image_token:
                return image_token
        return None

    def _prepare_image_inputs(
        self, processor: Any, pil_image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = self._processor_call(text="", images=[pil_image])
        input_ids = self._to_numpy(inputs["input_ids"])
        if input_ids.size == 0:
            image_token = self._find_image_token_text(processor)
            inputs = self._processor_call(
                text=image_token or "<|image_pad|>", images=[pil_image]
            )
            input_ids = self._to_numpy(inputs["input_ids"])

        if input_ids.size == 0:
            raise RuntimeError(
                "Processor returned empty input_ids for image embedding."
            )

        input_ids = input_ids.astype(np.int64, copy=False)
        pixel_values = self._to_numpy(inputs["pixel_values"])
        image_grid_thw = self._to_numpy(inputs["image_grid_thw"]).astype(
            np.int64, copy=False
        )
        return input_ids, pixel_values, image_grid_thw

    @staticmethod
    def _extract_vision_hidden(vision_out: Any) -> Any:
        if isinstance(vision_out, tuple):
            return vision_out[0]
        if hasattr(vision_out, "last_hidden_state"):
            return vision_out.last_hidden_state
        if hasattr(vision_out, "hidden_states"):
            return vision_out.hidden_states
        return vision_out

    def _image_mask_positions(
        self, input_ids: np.ndarray
    ) -> tuple[np.ndarray, list[int]]:
        image_mask = (input_ids == self._image_token_id)[0]
        image_positions = np.where(image_mask)[0].tolist()
        return image_mask, image_positions

    def _vision_embeddings(
        self,
        model: Any,
        pixel_values: np.ndarray,
        image_grid_thw: np.ndarray,
    ) -> np.ndarray:
        vision_tower = self._get_vision_tower(model)
        vision_out = vision_tower(
            mx.array(pixel_values), mx.array(image_grid_thw, dtype=mx.int64)
        )
        hidden_states = self._extract_vision_hidden(vision_out)
        return self._to_numpy(hidden_states)

    def _combine_text_vision_embeddings(
        self,
        model: Any,
        input_ids: np.ndarray,
        vision_np: np.ndarray,
        image_positions: list[int],
    ) -> tuple[Any, mx.array]:
        inner_model = self._get_language_model(model)
        embed_tokens = self._get_child(inner_model, "embed_tokens")
        if embed_tokens is None:
            raise RuntimeError("embed_tokens missing in qwen3_vl language model")

        input_ids_mx = mx.array(input_ids, dtype=mx.int64)
        text_emb = embed_tokens(input_ids_mx)
        text_emb_np = self._to_numpy(text_emb)[0]

        for i, pos in enumerate(image_positions):
            if i < vision_np.shape[0]:
                text_emb_np[pos] = vision_np[i]

        combined_embeddings = mx.array(text_emb_np).reshape(
            1, -1, text_emb_np.shape[-1]
        )
        return inner_model, combined_embeddings

    def _select_image_hidden(
        self, hidden_states: mx.array, image_mask: np.ndarray
    ) -> mx.array:
        hidden_states_np = self._to_numpy(hidden_states)
        return mx.array(hidden_states_np[0][image_mask])

    def _decode_image(self, image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, Path | str):
            path = Path(image)
            try:
                if path.exists():
                    return Image.open(str(path)).convert("RGB")
            except OSError:
                # Likely a long base64 string, fall through to base64 handling.
                pass

        if isinstance(image, str):
            data = image
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            image_bytes = base64.b64decode(data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        raise ValueError("Unsupported image input")

    def _embed_text_hidden_states(
        self,
        text: str,
    ) -> tuple[mx.array, mx.array, mx.array | None, int]:
        self.load()

        model, processor = self._get_backend()
        reset_request_local_runtime_state(model)

        tokenizer = getattr(self.tomoro_processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = processor

        inputs = self._tokenize_text(tokenizer, text)
        input_ids_np = self._to_numpy(inputs["input_ids"]).astype(np.int64, copy=False)
        input_ids = mx.array(input_ids_np, dtype=mx.int64)

        attention_mask = inputs.get("attention_mask")
        attention_mask_mx: mx.array | None = None
        if attention_mask is not None:
            attention_mask_np = self._to_numpy(attention_mask).astype(
                np.int32, copy=False
            )
            attention_mask_mx = mx.array(attention_mask_np, dtype=mx.int32)

        batch_size, seq_len = input_ids.shape

        inner_model = self._get_language_model(model)
        embed_tokens = self._get_child(inner_model, "embed_tokens")
        if embed_tokens is None:
            raise RuntimeError("embed_tokens missing in qwen3_vl language model")

        inputs_embeds = embed_tokens(input_ids)
        position_ids = self._build_position_ids(batch_size, seq_len)
        hidden_states = self._run_language_layers(
            inner_model, inputs_embeds, position_ids
        )
        token_count = self._count_text_tokens(input_ids, attention_mask_mx)
        return hidden_states, input_ids, attention_mask_mx, token_count

    def embed_text(self, text: str) -> EmbeddingResult:
        with vlm_execution(self.model_id):
            hidden_states, _, _, token_count = self._embed_text_hidden_states(text)
            embeddings = self._project_and_normalize(hidden_states).squeeze(0)
            mx.eval(embeddings)

            del hidden_states
            mx.clear_cache()

            return EmbeddingResult(
                embeddings=embeddings,
                num_tokens=token_count,
                source_type="text",
            )

    def embed_text_pooled(self, text: str) -> EmbeddingResult:
        """Return one sentence embedding from the shared VLM language tower."""
        with vlm_execution(self.model_id):
            hidden_states, _, attention_mask, token_count = (
                self._embed_text_hidden_states(text)
            )
            pooled = self._last_token_pool(hidden_states, attention_mask)
            embeddings = self._project_and_normalize(pooled).squeeze(0)
            mx.eval(embeddings)

            del hidden_states
            del attention_mask
            del pooled
            mx.clear_cache()

            return EmbeddingResult(
                embeddings=embeddings,
                num_tokens=token_count,
                source_type="text",
            )

    def embed_image(self, image: str | Path | Image.Image) -> EmbeddingResult:
        self.load()

        if not self._has_vision:
            raise RuntimeError(
                "qwen3_vl model has no vision assets. Use a vision-capable model."
            )

        with vlm_execution(self.model_id):
            model, processor = self._get_backend()
            reset_request_local_runtime_state(model)

            pil_image = self._decode_image(image)
            input_ids, pixel_values, image_grid_thw = self._prepare_image_inputs(
                processor, pil_image
            )
            image_mask, image_positions = self._image_mask_positions(input_ids)
            vision_np = self._vision_embeddings(model, pixel_values, image_grid_thw)
            inner_model, combined_embeddings = self._combine_text_vision_embeddings(
                model, input_ids, vision_np, image_positions
            )
            batch_size, seq_len, _ = combined_embeddings.shape
            position_ids = self._build_position_ids(batch_size, seq_len)

            hidden_states = self._run_language_layers(
                inner_model, combined_embeddings, position_ids
            )
            image_hidden_states = self._select_image_hidden(hidden_states, image_mask)
            embeddings = self._project_and_normalize(image_hidden_states)
            mx.eval(embeddings)

            del hidden_states
            del image_hidden_states
            del combined_embeddings
            del vision_np
            mx.clear_cache()

            return EmbeddingResult(
                embeddings=embeddings,
                num_tokens=embeddings.shape[0],
                source_type="image",
            )

    def embed_pdf(
        self,
        pdf_path: str | Path,
        dpi: int = 150,
        max_pages: int | None = None,
    ) -> list[EmbeddingResult]:
        if fitz is None:
            raise ImportError("PyMuPDF required: pip install pymupdf") from (
                _fitz_import_error
            )

        doc = fitz.open(str(pdf_path))
        num_pages = min(len(doc), max_pages) if max_pages else len(doc)
        results = []
        for i in range(num_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            results.append(self.embed_image(image))
        doc.close()
        return results

    @staticmethod
    def maxsim_score(query_embedding: Iterable, doc_embedding: Iterable) -> float:
        query_mx = mx.array(list(query_embedding))
        doc_mx = mx.array(list(doc_embedding))
        similarities = query_mx @ doc_mx.T
        max_sims = mx.max(similarities, axis=1)
        score = mx.sum(max_sims)
        mx.eval(score)
        return float(score)

    @staticmethod
    def to_numpy(emb: mx.array | EmbeddingResult) -> np.ndarray:
        if isinstance(emb, EmbeddingResult):
            emb = emb.embeddings
        return Qwen3VLEmbedder._to_numpy(emb)

    def log_summary(self) -> None:
        logger.info(
            "Qwen3VLEmbedder ready (model=%s, dim=%s, vision=%s)",
            self.model_id,
            self.embedding_dim,
            self._has_vision,
        )
