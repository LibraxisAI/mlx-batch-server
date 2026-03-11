from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_batch_server.chat.mlx import chat_generator as chat_generator_module
from mlx_batch_server.chat.mlx.chat_generator import ChatGenerator


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(1, len(text.split()) + 1))

    def decode(self, tokens: list[int]) -> str:
        return "".join(str(token) for token in tokens)


class _FakeChatTemplate:
    enable_thinking_parse = False

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tools=None,
        **kwargs,
    ) -> str:
        del tools, kwargs
        return messages[0]["content"]

    def stream_parse_chat_result(self, text: str):
        return SimpleNamespace(thinking=None, content=text)


def _fake_wrapper(context_length: int, *, multimodal: bool = False) -> ChatGenerator:
    text_model = object()
    runtime_model = (
        SimpleNamespace(language_model=text_model) if multimodal else text_model
    )
    model = SimpleNamespace(
        model_id="test-model",
        context_length=context_length,
        config={"max_position_embeddings": context_length},
        tokenizer=_FakeTokenizer(),
        chat_template=_FakeChatTemplate(),
        model=runtime_model,
        text_model=text_model,
        draft_model=None,
    )
    return ChatGenerator(model)


class TestChatGeneratorLimits:
    def test_resolve_max_tokens_clamps_to_remaining_context(self):
        """Sequential generation should clamp to the remaining context."""
        wrapper = _fake_wrapper(context_length=8)

        assert wrapper._resolve_max_tokens(requested=10, prompt_tokens=7) == 1

    def test_generate_stream_rejects_non_positive_max_tokens(self):
        """Invalid token budgets should fail with an explicit validation error."""
        wrapper = _fake_wrapper(context_length=8)

        with pytest.raises(RuntimeError, match="max_tokens must be a positive integer"):
            list(
                wrapper.generate_stream(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=0,
                )
            )

    def test_generate_stream_passes_clamped_budget_to_mlx(self, monkeypatch):
        """The MLX call should see a context-aware max_tokens budget."""
        wrapper = _fake_wrapper(context_length=8)
        state = {"kwargs": None}

        def fake_stream_generate(
            *, model, tokenizer, prompt, draft_model=None, **kwargs
        ):
            del model, tokenizer, prompt, draft_model
            state["kwargs"] = kwargs
            yield SimpleNamespace(
                token=1,
                finish_reason=None,
                prompt_tokens=7,
                generation_tokens=1,
                prompt_tps=10.0,
                generation_tps=20.0,
                peak_memory=1.0,
                from_draft=False,
            )
            yield SimpleNamespace(
                token=0,
                finish_reason="stop",
                prompt_tokens=7,
                generation_tokens=1,
                prompt_tps=10.0,
                generation_tps=20.0,
                peak_memory=1.0,
                from_draft=False,
            )

        monkeypatch.setattr(
            chat_generator_module,
            "stream_generate",
            fake_stream_generate,
        )

        results = list(
            wrapper.generate_stream(
                messages=[{"role": "user", "content": "a b c d e f g"}],
                max_tokens=10,
            )
        )

        assert len(results) == 1
        assert state["kwargs"]["max_tokens"] == 1

    def test_generate_stream_uses_language_model_for_vlm_runtime(self, monkeypatch):
        """Text-only requests on VLM runtimes should use the language tower."""
        wrapper = _fake_wrapper(context_length=8, multimodal=True)
        state = {"model": None, "kwargs": None}

        def fake_stream_generate(
            *, model, tokenizer, prompt, draft_model=None, **kwargs
        ):
            del tokenizer, prompt, draft_model
            state["model"] = model
            state["kwargs"] = kwargs
            yield SimpleNamespace(
                token=1,
                finish_reason=None,
                prompt_tokens=7,
                generation_tokens=1,
                prompt_tps=10.0,
                generation_tps=20.0,
                peak_memory=1.0,
                from_draft=False,
            )
            yield SimpleNamespace(
                token=0,
                finish_reason="stop",
                prompt_tokens=7,
                generation_tokens=1,
                prompt_tps=10.0,
                generation_tps=20.0,
                peak_memory=1.0,
                from_draft=False,
            )

        monkeypatch.setattr(
            chat_generator_module,
            "stream_generate",
            fake_stream_generate,
        )

        results = list(
            wrapper.generate_stream(
                messages=[{"role": "user", "content": "a b c d e f g"}],
                max_tokens=10,
            )
        )

        assert len(results) == 1
        assert state["model"] is wrapper.model.text_model
        assert state["kwargs"]["max_tokens"] == 1
