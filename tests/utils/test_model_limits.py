from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_batch_server.utils.model_limits import (
    extract_context_length,
    resolve_max_tokens,
)


def test_extract_context_length_reads_nested_config():
    """Nested text config should win when present."""
    config = {
        "text_config": {
            "max_position_embeddings": 32768,
        }
    }

    assert extract_context_length(config) == 32768


def test_extract_context_length_prefers_larger_nested_text_window():
    """Nested text towers should override smaller top-level multimodal values."""
    config = {
        "max_position_embeddings": 8192,
        "text_config": {
            "max_position_embeddings": 32768,
        },
    }

    assert extract_context_length(config) == 32768


def test_extract_context_length_uses_rope_scaling():
    """RoPE scaling metadata should expand the effective context length."""
    config = {
        "max_position_embeddings": 8192,
        "rope_scaling": {
            "factor": 4,
            "original_max_position_embeddings": 8192,
        },
    }

    assert extract_context_length(config) == 32768


def test_extract_context_length_falls_back_to_tokenizer():
    """Tokenizer model_max_length should be used when config is inconclusive."""
    tokenizer = SimpleNamespace(model_max_length=16384)

    assert extract_context_length({}, tokenizer) == 16384


def test_resolve_max_tokens_clamps_to_remaining_context():
    """Requested output should never exceed the remaining context window."""
    assert (
        resolve_max_tokens(
            requested=4000,
            context_length=4096,
            prompt_tokens=3000,
        )
        == 1096
    )


def test_resolve_max_tokens_defaults_to_remaining_context():
    """Missing max_tokens should use the remaining context budget."""
    assert (
        resolve_max_tokens(
            requested=None,
            context_length=4096,
            prompt_tokens=1024,
        )
        == 3072
    )


def test_resolve_max_tokens_uses_fallback_for_unknown_context():
    """Unknown context should use the configured fallback when no request is set."""
    assert (
        resolve_max_tokens(
            requested=None,
            context_length=None,
            fallback=2048,
            context_label="test-model",
        )
        == 2048
    )


def test_resolve_max_tokens_raises_when_no_limits_are_available():
    """No request plus unknown context and no fallback should fail loudly."""
    with pytest.raises(ValueError):
        resolve_max_tokens(
            requested=None,
            context_length=None,
            fallback=None,
        )
