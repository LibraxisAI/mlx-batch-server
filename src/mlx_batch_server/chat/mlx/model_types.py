"""MLX Model types and management."""

import importlib
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import KVCache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load as load_text_runtime
from mlx_lm.utils import load_config as load_text_config

# Handle mlx_lm version compatibility: newer versions use hf_repo_to_path (returns Path),
# older versions (< 0.29) use get_model_path (returns Tuple[Path, Optional[str]])
try:
    from mlx_lm.utils import hf_repo_to_path as _get_model_path

    def get_model_path(model_id: str) -> Path:
        """Get model path (wrapper for newer mlx_lm versions)."""
        return _get_model_path(model_id)

except ImportError:
    from mlx_lm.utils import get_model_path as _get_model_path

    def get_model_path(model_id: str) -> Path:
        """Get model path (wrapper for older mlx_lm versions)."""
        result = _get_model_path(model_id)
        # Old version returns Tuple[Path, Optional[str]], extract just the Path
        if isinstance(result, tuple):
            return result[0]
        elif isinstance(result, Path):
            return result
        else:
            raise TypeError(
                f"Unexpected return type from get_model_path: {type(result)}"
            )


from ...utils.logger import logger
from ...utils.model_limits import extract_context_length
from .runtime_aliases import resolve_runtime_model_id
from .tools.chat_template import ChatTemplate

_MULTIMODAL_CONFIG_KEYS = (
    "vision_config",
    "vision_model",
    "vision_tower",
    "image_token_id",
    "image_token_index",
    "video_token_id",
    "audio_config",
    "audio_tower",
)
_MULTIMODAL_MODEL_TYPE_MARKERS = (
    "vl",
    "vision",
    "llava",
    "idefics",
    "molmo",
    "bunny",
)


class _MLXLMCompatibleCacheProxy:
    """Normalize mlx_lm batch cache offsets for VLM language towers."""

    def __init__(self, base_cache: Any):
        self.base_cache = base_cache

    @property
    def offset(self):
        offset = self.base_cache.offset
        if isinstance(offset, int):
            return offset
        if isinstance(offset, mx.array):
            if offset.ndim == 0:
                return int(offset.item())
            batch_index = getattr(self.base_cache, "_idx", None)
            if batch_index is not None:
                return int(batch_index)
            if offset.size == 1:
                return int(offset[0].item())
            return int(offset.max().item())
        return offset

    @offset.setter
    def offset(self, value):
        self.base_cache.offset = value

    def __getattr__(self, name: str):
        return getattr(self.base_cache, name)

    def __getitem__(self, idx):
        return self.base_cache[idx]

    def __setitem__(self, idx, value):
        self.base_cache[idx] = value


class MLXLMCompatibleLanguageModel(nn.Module):
    """Adapt VLM language towers to the logits tensor contract used by mlx_lm.

    ``mlx_lm`` generation helpers expect ``model(...)`` to return a logits tensor
    shaped like ``[batch, seq, vocab]``. ``mlx_vlm`` language towers instead
    return an object with a ``.logits`` attribute. This wrapper normalizes that
    forward contract while preserving cache-related attributes needed for batch
    and prompt-cache flows.
    """

    def __init__(self, base_model: Any):
        super().__init__()
        self.base_model = base_model

    def _normalize_cache(self, cache):
        if cache is None:
            return None
        normalized = []
        for entry in cache:
            if entry is None or not hasattr(entry, "offset"):
                normalized.append(entry)
                continue
            normalized.append(_MLXLMCompatibleCacheProxy(entry))
        return normalized

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        **kwargs,
    ):
        if input_embeddings is not None and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = input_embeddings

        output = self.base_model(inputs, cache=self._normalize_cache(cache), **kwargs)
        return getattr(output, "logits", output)

    def make_cache(self):
        if hasattr(self.base_model, "make_cache"):
            return self.base_model.make_cache()
        if hasattr(self.base_model, "layers"):
            return [KVCache() for _ in self.base_model.layers]
        raise AttributeError("Wrapped language model does not define make_cache()")

    @property
    def layers(self):
        return self.base_model.layers

    @property
    def head_dim(self):
        return self.base_model.head_dim

    @property
    def n_kv_heads(self):
        return self.base_model.n_kv_heads

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "base_model"), name)


def _fix_tokenizer_eos(tokenizer: TokenizerWrapper) -> None:
    """Fix tokenizer eos_token_ids to include the actual eos_token.

    Some models have mismatched config.json eos_token_id and tokenizer eos_token.
    For example, Nemotron has config.json with eos_token_id=2, but the actual
    eos_token '<|im_end|>' has ID 11. This causes mlx_lm to not stop generation
    on the correct token.

    This function ensures eos_token_ids contains the ID of the actual eos_token.
    """
    if not hasattr(tokenizer, "_tokenizer") or not hasattr(tokenizer, "_eos_token_ids"):
        return

    underlying = tokenizer._tokenizer
    eos_token = getattr(underlying, "eos_token", None)

    if not eos_token:
        return

    # Get the actual token ID from vocabulary
    vocab = underlying.get_vocab()
    actual_eos_id = vocab.get(eos_token)

    if actual_eos_id is not None and actual_eos_id not in tokenizer._eos_token_ids:
        tokenizer._eos_token_ids.add(actual_eos_id)
        logger.info(
            f"Fixed eos_token_ids: added {eos_token!r} (ID {actual_eos_id}) "
            f"-> eos_token_ids now: {tokenizer._eos_token_ids}"
        )


def _wrap_tokenizer(tokenizer_like: Any) -> TokenizerWrapper:
    if isinstance(tokenizer_like, TokenizerWrapper):
        return tokenizer_like
    return TokenizerWrapper(tokenizer_like)


def _is_local_path(model_id: str) -> bool:
    """Check if model_id is a local filesystem path."""
    return (
        model_id.startswith("/")
        or model_id.startswith("~")
        or model_id.startswith("./")
    )


def _looks_multimodal_config(config: dict[str, Any]) -> bool:
    if any(config.get(key) is not None for key in _MULTIMODAL_CONFIG_KEYS):
        return True

    model_type = str(config.get("model_type", "")).lower()
    if any(marker in model_type for marker in _MULTIMODAL_MODEL_TYPE_MARKERS):
        return True

    architectures = config.get("architectures") or []
    return any(
        any(
            marker in str(architecture).lower()
            for marker in _MULTIMODAL_MODEL_TYPE_MARKERS
        )
        for architecture in architectures
    )


def _should_use_vlm_runtime(config: dict[str, Any]) -> bool:
    if not _looks_multimodal_config(config):
        return False

    try:
        get_vlm_model_and_args = importlib.import_module(
            "mlx_vlm.utils"
        ).get_model_and_args
    except Exception as exc:
        raise RuntimeError(
            "mlx-vlm is required to load vision-language models"
        ) from exc

    try:
        get_vlm_model_and_args(config)
    except Exception:
        return False

    return True


def _load_vlm_runtime(
    model_id: str,
    adapter_path: str | None,
) -> tuple[nn.Module, TokenizerWrapper, Any, Any]:
    try:
        load_vlm = importlib.import_module("mlx_vlm").load
    except Exception as exc:
        raise RuntimeError(
            "mlx-vlm is required to load vision-language models"
        ) from exc

    model, processor = load_vlm(model_id, adapter_path=adapter_path)
    tokenizer_source = getattr(processor, "tokenizer", processor)
    tokenizer = _wrap_tokenizer(tokenizer_source)
    _fix_tokenizer_eos(tokenizer)
    chat_template_source = (
        processor if hasattr(processor, "apply_chat_template") else tokenizer
    )
    return model, tokenizer, processor, chat_template_source


def load_mlx_model(
    model_id: str,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> "MLXModel":
    """Factory function to load MLX models.

    Args:
        model_id: Model name/path (HuggingFace model ID or local path)
        adapter_path: Optional path to LoRA adapter
        draft_model_id: Optional draft model name/path for speculative decoding

    Returns:
        MLXModel instance with loaded models

    Raises:
        ValueError: If model_id is invalid
        RuntimeError: If model loading fails
    """
    if not model_id or not model_id.strip():
        raise ValueError("model_id cannot be empty")

    model_id = resolve_runtime_model_id(model_id).strip()
    if draft_model_id:
        draft_model_id = resolve_runtime_model_id(draft_model_id).strip()

    # Expand home directory if needed
    if model_id.startswith("~"):
        model_id = str(Path(model_id).expanduser())
    if draft_model_id and draft_model_id.startswith("~"):
        draft_model_id = str(Path(draft_model_id).expanduser())

    try:
        # Load configuration - use path directly for local models
        if _is_local_path(model_id):
            model_path = Path(model_id)
        else:
            model_path = get_model_path(model_id)
        config = load_text_config(model_path)
        processor = None

        if _should_use_vlm_runtime(config):
            model, tokenizer, processor, chat_template_source = _load_vlm_runtime(
                model_id=model_id,
                adapter_path=adapter_path,
            )
            logger.info("Loaded VLM runtime for text generation: %s", model_id)
        else:
            model, tokenizer = load_text_runtime(
                model_id,
                tokenizer_config={"trust_remote_code": True},
                adapter_path=adapter_path,
            )
            tokenizer = _wrap_tokenizer(tokenizer)
            _fix_tokenizer_eos(tokenizer)
            chat_template_source = tokenizer
            logger.info(f"Loaded model: {model_id}")

        chat_template = ChatTemplate(config["model_type"], chat_template_source)

        # Load draft model if specified
        draft_model = None
        draft_tokenizer = None
        if draft_model_id:
            try:
                draft_model, draft_tokenizer = load_text_runtime(
                    draft_model_id,
                    tokenizer_config={"trust_remote_code": True},
                )
                draft_tokenizer = _wrap_tokenizer(draft_tokenizer)
                # Fix potential eos_token_ids mismatch for draft model
                _fix_tokenizer_eos(draft_tokenizer)

                # Check if vocabulary sizes match
                if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                    logger.warn(
                        f"Draft model({draft_model_id}) tokenizer does not match model tokenizer."
                    )

                logger.info(f"Loaded draft model: {draft_model_id}")
            except Exception as e:
                logger.error(f"Failed to load draft model {draft_model_id}: {e}")
                # Continue without draft model
                draft_model = None
                draft_tokenizer = None

        return MLXModel(
            model_id=model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
            config=config,
            model=model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            processor=processor,
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
        )

    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise RuntimeError(f"Model loading failed for {model_id}: {e}") from e


class MLXModel:
    """Simplified MLX model container.

    This class is a simple data container for loaded MLX models.
    For model management operations, create new instances rather than modifying existing ones.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None,
        draft_model_id: str | None,
        config: dict,
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        chat_template: ChatTemplate,
        processor: Any | None = None,
        draft_model: nn.Module | None = None,
        draft_tokenizer: TokenizerWrapper | None = None,
    ):
        """Initialize MLX model container.

        This constructor is typically called by load_mlx_model() factory function.

        Args:
            model_id: Model name/path
            adapter_path: Path to LoRA adapter (if any)
            draft_model_id: Draft model name/path (if any)
            config: Loaded model configuration
            model: Loaded main model
            tokenizer: Loaded tokenizer
            chat_template: Chat template instance
            draft_model: Loaded draft model (optional)
            draft_tokenizer: Draft model tokenizer (optional)
        """
        # Model identification
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.draft_model_id = draft_model_id
        self.config = config
        self.context_length = extract_context_length(config, tokenizer)

        # Loaded model components
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.processor = processor
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self._text_model_proxy: nn.Module | None = None

    def reset_runtime_state(self) -> None:
        """Clear request-local transient state left behind by some VLM towers."""
        candidate = getattr(self.model, "language_model", self.model)
        cleared = False

        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(candidate, attr):
                setattr(candidate, attr, None)
                cleared = True

        if cleared:
            logger.debug("Cleared transient runtime state for %s", self.model_id)

    @property
    def text_model(self) -> nn.Module:
        """Return the text-generation tower for both LLM and VLM runtimes."""
        base_model = getattr(self.model, "language_model", self.model)
        if not self.supports_multimodal:
            return base_model

        if self._text_model_proxy is None:
            self._text_model_proxy = MLXLMCompatibleLanguageModel(base_model)
        return self._text_model_proxy

    @property
    def supports_multimodal(self) -> bool:
        """Whether this runtime also carries the full VLM processor stack."""
        return self.processor is not None

    def new_chat_template(self) -> ChatTemplate:
        """Return a fresh request-local ChatTemplate instance."""
        if hasattr(self.chat_template, "fork"):
            return self.chat_template.fork()
        return self.chat_template

    @classmethod
    def load(
        cls,
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> "MLXModel":
        return load_mlx_model(model_id, adapter_path, draft_model_id)

    def __str__(self) -> str:
        """Return a string representation of the model for debugging."""
        parts = [f"model_id={self.model_id}"]
        if self.adapter_path:
            parts.append(f"adapter_path={self.adapter_path}")
        if self.draft_model_id:
            parts.append(f"draft_model_id={self.draft_model_id}")
        return f"MLXModel({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        """Check equality based on model configuration."""
        if not isinstance(other, MLXModel):
            return False
        return (
            self.model_id == other.model_id
            and self.adapter_path == other.adapter_path
            and self.draft_model_id == other.draft_model_id
        )

    def __hash__(self) -> int:
        """Hash based on model configuration for use as dict keys."""
        return hash((self.model_id, self.adapter_path, self.draft_model_id))

    def has_adapter(self) -> bool:
        """Check if this model has an adapter configured."""
        return self.adapter_path is not None

    def has_draft_model(self) -> bool:
        """Check if draft model is available."""
        return self.draft_model is not None and self.draft_tokenizer is not None
