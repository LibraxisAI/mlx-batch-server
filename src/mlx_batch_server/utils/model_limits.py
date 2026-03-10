from __future__ import annotations

from typing import Any

from .logger import logger

_CONTEXT_LENGTH_KEYS = (
    "max_position_embeddings",
    "max_seq_len",
    "max_sequence_length",
    "model_max_length",
    "n_positions",
    "seq_length",
    "context_length",
)

_UNKNOWN_CONTEXT_FALLBACK_MAX_TOKENS = 4096


def _coerce_positive_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return None

    if int_value <= 0:
        return None

    # Hugging Face uses huge sentinels when the true limit is unknown.
    if int_value >= 1_000_000_000:
        return None

    return int_value


def _get_mapping_value(obj: Any, key: str) -> Any | None:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_context_length(config: Any, tokenizer: Any | None = None) -> int | None:
    candidates: list[Any] = []
    if config is not None:
        candidates.append(config)

        for nested_key in (
            "text_config",
            "language_config",
            "model_config",
            "llm_config",
            "config",
            "language_model",
        ):
            nested = _get_mapping_value(config, nested_key)
            if nested is not None:
                candidates.append(nested)

    for candidate in candidates:
        direct_values: list[int] = []
        for key in _CONTEXT_LENGTH_KEYS:
            value = _coerce_positive_int(_get_mapping_value(candidate, key))
            if value is not None:
                direct_values.append(value)

        rope_scaling = _get_mapping_value(candidate, "rope_scaling")
        if isinstance(rope_scaling, dict):
            factor = _coerce_positive_int(rope_scaling.get("factor"))
            base = _coerce_positive_int(
                rope_scaling.get("original_max_position_embeddings")
            ) or _coerce_positive_int(
                _get_mapping_value(candidate, "max_position_embeddings")
            )
            if factor and base:
                direct_values.append(base * factor)

        if direct_values:
            return max(direct_values)

    if tokenizer is not None:
        tok_length = _coerce_positive_int(getattr(tokenizer, "model_max_length", None))
        if tok_length is not None:
            return tok_length

    return None


def resolve_max_tokens(
    *,
    requested: int | None,
    context_length: int | None,
    prompt_tokens: int | None = None,
    fallback: int | None = _UNKNOWN_CONTEXT_FALLBACK_MAX_TOKENS,
    context_label: str = "model",
) -> int:
    requested_tokens = _coerce_positive_int(requested)

    if context_length is not None:
        available = context_length
        if prompt_tokens is not None:
            available = max(1, context_length - prompt_tokens)

        if requested_tokens is None:
            return available

        return min(requested_tokens, available)

    if requested_tokens is not None:
        return requested_tokens

    if fallback is None:
        raise ValueError("No max_tokens specified and context length is unknown")

    logger.warning(
        "Context length unknown for %s; falling back to max_tokens=%s",
        context_label,
        fallback,
    )
    return fallback
