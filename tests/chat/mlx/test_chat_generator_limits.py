from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_batch_server.chat.mlx import chat_generator as chat_generator_module
from mlx_batch_server.chat.mlx import runtime_attachments as runtime_attachments_module
from mlx_batch_server.chat.mlx.chat_generator import ChatGenerator
from mlx_batch_server.chat.mlx.model_types import (
    MLXLMCompatibleLanguageModel,
    MLXModel,
)
from mlx_batch_server.chat.mlx.wrapper_cache import (
    wrapper_cache as shared_wrapper_cache,
)


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


class _FakeRuntimeTokenizer:
    model_max_length = 16


class _FakeLanguageTowerOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeVLMLanguageTower:
    layers = [object()]
    head_dim = 8
    n_kv_heads = 1

    def __init__(self):
        self.calls = []

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        self.calls.append(
            {
                "inputs": inputs,
                "cache": cache,
                "inputs_embeds": inputs_embeds,
                "kwargs": kwargs,
            }
        )
        batch, seq = inputs.shape
        logits = mx.array(
            [[[0.0, 1.0, -1.0, -2.0] for _ in range(seq)] for _ in range(batch)]
        )
        return _FakeLanguageTowerOutput(logits=logits)

    def make_cache(self):
        return []


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
        supports_multimodal=multimodal,
    )
    return ChatGenerator(model)


class TestChatGeneratorLimits:
    def setup_method(self):
        runtime_attachments_module.clear_runtime_surface_attachments()

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

    def test_generate_stream_preserves_stateful_unicode_and_terminal_text(
        self,
        monkeypatch,
    ):
        wrapper = _fake_wrapper(context_length=16)

        def fake_stream_generate(**kwargs):
            del kwargs
            common = {
                "prompt_tokens": 1,
                "prompt_tps": 10.0,
                "generation_tps": 20.0,
                "peak_memory": 1.0,
                "from_draft": False,
            }
            yield SimpleNamespace(
                token=1,
                text="Cześć! ",
                finish_reason=None,
                generation_tokens=1,
                **common,
            )
            yield SimpleNamespace(
                token=2,
                text="😊",
                finish_reason="stop",
                generation_tokens=2,
                **common,
            )

        monkeypatch.setattr(
            chat_generator_module,
            "stream_generate",
            fake_stream_generate,
        )

        results = list(
            wrapper.generate_stream(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=4,
            )
        )
        text = "".join(result.content.text_delta or "" for result in results)

        assert text == "Cześć! 😊"
        assert "\ufffd" not in text
        assert results[-1].finish_reason == "stop"

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

    def test_generate_stream_serializes_shared_vlm_runtime(self, monkeypatch):
        """VLM text requests should acquire the shared multimodal execution lock."""
        wrapper = _fake_wrapper(context_length=8, multimodal=True)
        events: list[tuple[str, str, str | None, str | None]] = []

        @contextmanager
        def fake_execution(
            model_id: str,
            *,
            adapter_path: str | None = None,
            draft_model_id: str | None = None,
        ):
            events.append(("enter", model_id, adapter_path, draft_model_id))
            try:
                yield
            finally:
                events.append(("exit", model_id, adapter_path, draft_model_id))

        def fake_stream_generate(
            *, model, tokenizer, prompt, draft_model=None, **kwargs
        ):
            del model, tokenizer, prompt, draft_model, kwargs
            assert events == [("enter", "test-model", None, None)]
            yield SimpleNamespace(
                token=1,
                finish_reason=None,
                prompt_tokens=2,
                generation_tokens=1,
                prompt_tps=10.0,
                generation_tps=20.0,
                peak_memory=1.0,
                from_draft=False,
            )
            yield SimpleNamespace(
                token=0,
                finish_reason="stop",
                prompt_tokens=2,
                generation_tokens=1,
                prompt_tps=10.0,
                generation_tps=20.0,
                peak_memory=1.0,
                from_draft=False,
            )

        monkeypatch.setattr(shared_wrapper_cache, "vlm_execution", fake_execution)
        monkeypatch.setattr(
            chat_generator_module,
            "stream_generate",
            fake_stream_generate,
        )

        results = list(
            wrapper.generate_stream(
                messages=[{"role": "user", "content": "hello world"}],
                max_tokens=4,
            )
        )

        assert len(results) == 1
        assert events == [
            ("enter", "test-model", None, None),
            ("exit", "test-model", None, None),
        ]

    def test_get_or_create_requests_llm_runtime_surface(self, monkeypatch):
        """Lazy text-model access should request llm retention from wrapper cache."""
        wrapper = _fake_wrapper(context_length=8, multimodal=True)
        wrapper.model.model_id = "LibraxisAI/Qwen3-VL-30B"
        seen_calls: list[tuple[str, str | None, str | None, str | None]] = []

        def fake_get_wrapper(
            model_id,
            adapter_path=None,
            draft_model_id=None,
            surface=None,
        ):
            seen_calls.append((model_id, adapter_path, draft_model_id, surface))
            return wrapper

        monkeypatch.setattr(
            shared_wrapper_cache,
            "get_wrapper",
            fake_get_wrapper,
        )

        loaded = ChatGenerator.get_or_create("frontier-vlm")

        assert loaded is wrapper
        assert seen_calls == [("frontier-vlm", None, None, "llm")]

    def test_vlm_text_model_is_mlx_lm_compatible_with_real_generate_step(self):
        """The VLM seam should unwrap LanguageModelOutput for real mlx_lm generation."""
        tower = _FakeVLMLanguageTower()
        model = MLXModel(
            model_id="test-vlm",
            adapter_path=None,
            draft_model_id=None,
            config={"max_position_embeddings": 16},
            model=SimpleNamespace(language_model=tower),
            tokenizer=_FakeRuntimeTokenizer(),
            chat_template=_FakeChatTemplate(),
            processor=object(),
        )

        text_model = model.text_model

        assert isinstance(text_model, MLXLMCompatibleLanguageModel)

        from mlx_lm.generate import generate_step

        token, logprobs = next(
            generate_step(
                prompt=mx.array([1]),
                model=text_model,
                max_tokens=1,
                prompt_cache=[],
            )
        )

        assert token == 1
        assert logprobs.shape == (4,)
        assert tower.calls[0]["cache"] == []
        assert text_model.make_cache() == []
