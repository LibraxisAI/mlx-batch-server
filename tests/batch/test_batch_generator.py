from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_batch_server.batch import generator as batch_generator_module
from mlx_batch_server.batch.generator import BatchChatGenerator, BatchRequest


class _FakeTokenizer:
    eos_token_ids = {0}

    def encode(self, text: str) -> list[int]:
        return list(range(1, len(text.split()) + 1))

    def decode(self, tokens: list[int]) -> str:
        return "".join(str(token) for token in tokens)


class _FakeChatTemplate:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tools=None,
        **kwargs,
    ) -> str:
        del tools, kwargs
        return messages[0]["content"]


def _fake_model(context_length: int) -> SimpleNamespace:
    text_model = object()
    return SimpleNamespace(
        model_id="test-model",
        config={"max_position_embeddings": context_length},
        tokenizer=_FakeTokenizer(),
        chat_template=_FakeChatTemplate(),
        model=text_model,
        text_model=text_model,
    )


class TestBatchChatGenerator:
    def test_prepare_batch_requests_resolves_max_tokens_against_context(self):
        """Requests should be clamped to the remaining context budget."""
        generator = BatchChatGenerator(model=_fake_model(context_length=8))

        prompts, max_tokens, samplers = generator._prepare_batch_requests(
            [
                BatchRequest(
                    id="req-default",
                    messages=[{"role": "user", "content": "a b c"}],
                    max_tokens=None,
                ),
                BatchRequest(
                    id="req-clamped",
                    messages=[{"role": "user", "content": "a b c d e f g"}],
                    max_tokens=10,
                ),
            ]
        )

        assert prompts == [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7]]
        assert max_tokens == [5, 1]
        assert samplers == [None, None]

    @pytest.mark.asyncio
    async def test_stream_batch_passes_per_request_samplers(self, monkeypatch):
        """Concurrent requests should retain independent sampler settings."""
        generator = BatchChatGenerator(model=_fake_model(context_length=16))

        class FakeBatchGenerator:
            def __init__(self):
                self.insert_calls = []
                self._next_calls = 0

            def insert(self, prompts, max_tokens, samplers=None):
                self.insert_calls.append(
                    {
                        "prompts": prompts,
                        "max_tokens": max_tokens,
                        "samplers": samplers,
                    }
                )
                return [11, 22]

            def next(self):
                if self._next_calls > 0:
                    return []
                self._next_calls += 1
                return [
                    SimpleNamespace(
                        uid=11,
                        token=1,
                        logprobs=None,
                        finish_reason="stop",
                    ),
                    SimpleNamespace(
                        uid=22,
                        token=2,
                        logprobs=None,
                        finish_reason="stop",
                    ),
                ]

            def stats(self):
                return SimpleNamespace(
                    prompt_tokens=5,
                    generation_tokens=2,
                    prompt_tps=10.0,
                    generation_tps=20.0,
                    peak_memory=1.5,
                )

        fake_batch_gen = FakeBatchGenerator()
        generator._generator = fake_batch_gen

        monkeypatch.setattr(
            batch_generator_module,
            "make_sampler",
            lambda **kwargs: {"built": kwargs},
        )
        monkeypatch.setattr(
            generator,
            "_get_or_create_generator",
            lambda max_tokens: fake_batch_gen,
        )

        chunks = []
        async for chunk in generator.stream_batch(
            [
                BatchRequest(
                    id="req-greedy",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=3,
                ),
                BatchRequest(
                    id="req-temp",
                    messages=[{"role": "user", "content": "hello world"}],
                    max_tokens=4,
                    sampler_config={"temp": 0.7},
                ),
            ]
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert fake_batch_gen.insert_calls[0]["max_tokens"] == [3, 4]
        assert fake_batch_gen.insert_calls[0]["samplers"] == [
            None,
            {"built": {"temp": 0.7}},
        ]

    def test_get_or_create_generator_uses_language_model_for_vlm_runtime(
        self, monkeypatch
    ):
        """VLM-backed wrappers should batch on the text tower only."""
        language_model = object()
        generator = BatchChatGenerator(
            model=SimpleNamespace(
                model_id="test-vlm",
                config={"text_config": {"max_position_embeddings": 16}},
                tokenizer=_FakeTokenizer(),
                chat_template=_FakeChatTemplate(),
                model=SimpleNamespace(language_model=language_model),
                text_model=language_model,
            )
        )
        captured = {}

        class FakeBatchGenerator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            batch_generator_module,
            "BatchGenerator",
            FakeBatchGenerator,
        )

        generator._get_or_create_generator(max_tokens=4)

        assert captured["model"] is language_model
        assert captured["stop_tokens"] == {0}
