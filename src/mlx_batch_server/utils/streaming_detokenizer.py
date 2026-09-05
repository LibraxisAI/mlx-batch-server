"""Per-request streaming detokenizer helpers."""

from __future__ import annotations

from typing import Any

from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer, TokenizerWrapper


def new_streaming_detokenizer(tokenizer: Any) -> Any:
    """Return a fresh detokenizer that buffers incomplete UTF-8 token bytes."""

    detokenizer = (
        tokenizer.detokenizer
        if isinstance(tokenizer, TokenizerWrapper)
        else NaiveStreamingDetokenizer(tokenizer)
    )
    detokenizer.reset()
    return detokenizer


def push_token(detokenizer: Any, token: int) -> str:
    """Add one token and return only newly readable Unicode text."""

    detokenizer.add_token(token)
    return str(detokenizer.last_segment)


def finalize_text(detokenizer: Any) -> str:
    """Flush a request detokenizer and return its final readable segment."""

    detokenizer.finalize()
    return str(detokenizer.last_segment)
