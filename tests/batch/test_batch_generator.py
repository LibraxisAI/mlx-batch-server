from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mlx_batch_server.batch import generator as batch_generator_module
from mlx_batch_server.batch.generator import BatchChatGenerator, BatchRequest
from mlx_batch_server.chat.mlx.model_types import (
    MLXLMCompatibleLanguageModel,
    MLXModel,
)
from mlx_batch_server.chat.mlx.wrapper_cache import (
    wrapper_cache as shared_wrapper_cache,
)


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


class _FakeRuntimeTokenizer:
    model_max_length = 16


class _FakeLanguageTowerOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeVLMLanguageTower:
    layers = [object()]
    head_dim = 8
    n_kv_heads = 1

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        del cache, inputs_embeds, kwargs
        batch, seq = inputs.shape
        logits = mx.array(
            [[[0.0, 1.0, -1.0, -2.0] for _ in range(seq)] for _ in range(batch)]
        )
        return _FakeLanguageTowerOutput(logits=logits)

    def make_cache(self):
        return []


class _FakeBatchOffsetSensitiveTower:
    layers = [object()]
    head_dim = 8
    n_kv_heads = 1

    def __init__(self):
        self.seen_offset_types = []

    def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
        del inputs_embeds, kwargs
        if cache and cache[0] is not None:
            offset = cache[0].offset
            self.seen_offset_types.append(type(offset))
            kv_seq_len = inputs.shape[1] + offset + 1
            mask = mx.ones((inputs.shape[0], 1, inputs.shape[1], 32), dtype=mx.bool_)
            mask = mask[..., :kv_seq_len]
            assert mask.shape[-1] >= inputs.shape[1]
            head_shape = (inputs.shape[0], 1, inputs.shape[1], 1)
            cache[0].update_and_fetch(
                mx.zeros(head_shape, dtype=mx.float32),
                mx.zeros(head_shape, dtype=mx.float32),
            )

        batch, seq = inputs.shape
        logits = mx.array(
            [[[0.0, 1.0, -1.0, -2.0] for _ in range(seq)] for _ in range(batch)]
        )
        return _FakeLanguageTowerOutput(logits=logits)

    def make_cache(self):
        return [KVCache()]


def _fake_model(context_length: int, *, multimodal: bool = False) -> SimpleNamespace:
    text_model = object()
    runtime_model = (
        SimpleNamespace(language_model=text_model) if multimodal else text_model
    )
    return SimpleNamespace(
        model_id="test-model",
        config={"max_position_embeddings": context_length},
        tokenizer=_FakeTokenizer(),
        chat_template=_FakeChatTemplate(),
        model=runtime_model,
        text_model=text_model,
        supports_multimodal=multimodal,
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

    @pytest.mark.asyncio
    async def test_stream_batch_survives_zero_division_in_stats(self, monkeypatch):
        """A stats snapshot failure must not turn a successful batch into an error."""
        generator = BatchChatGenerator(model=_fake_model(context_length=16))

        class FakeBatchGenerator:
            def __init__(self):
                self._stats = SimpleNamespace(
                    prompt_tokens=4,
                    prompt_time=0.0,
                    generation_tokens=1,
                    generation_time=0.0,
                    peak_memory=1.25,
                )
                self._next_calls = 0

            def insert(self, prompts, max_tokens, samplers=None):
                del prompts, max_tokens, samplers
                return [11]

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
                    )
                ]

            def stats(self):
                raise ZeroDivisionError("division by zero")

        fake_batch_gen = FakeBatchGenerator()
        generator._generator = fake_batch_gen

        monkeypatch.setattr(
            generator,
            "_get_or_create_generator",
            lambda max_tokens: fake_batch_gen,
        )

        chunks = []
        async for chunk in generator.stream_batch(
            [
                BatchRequest(
                    id="req-zero-stats",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=2,
                )
            ]
        ):
            chunks.append(chunk)

        stats = generator.stats()

        assert len(chunks) == 1
        assert chunks[0].finish_reason == "stop"
        assert stats.prompt_tokens == 4
        assert stats.generation_tokens == 1
        assert stats.prompt_tps == 0.0
        assert stats.generation_tps == 0.0
        assert stats.peak_memory_gb == 1.25

    @pytest.mark.asyncio
    async def test_stream_batch_serializes_shared_vlm_runtime(self, monkeypatch):
        """VLM text batches should acquire the shared multimodal execution lock."""
        generator = BatchChatGenerator(
            model=_fake_model(context_length=16, multimodal=True)
        )
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

        class FakeBatchGenerator:
            def insert(self, prompts, max_tokens, samplers=None):
                del prompts, max_tokens, samplers
                assert events == [("enter", "test-model", None, None)]
                return [11]

            def next(self):
                assert events == [("enter", "test-model", None, None)]
                return [
                    SimpleNamespace(
                        uid=11,
                        token=1,
                        logprobs=None,
                        finish_reason="stop",
                    )
                ]

            def stats(self):
                return SimpleNamespace(
                    prompt_tokens=3,
                    generation_tokens=1,
                    prompt_tps=10.0,
                    generation_tps=20.0,
                    peak_memory=1.0,
                )

        fake_batch_gen = FakeBatchGenerator()
        generator._generator = fake_batch_gen

        monkeypatch.setattr(shared_wrapper_cache, "vlm_execution", fake_execution)
        monkeypatch.setattr(
            generator,
            "_get_or_create_generator",
            lambda max_tokens: fake_batch_gen,
        )

        chunks = []
        async for chunk in generator.stream_batch(
            [
                BatchRequest(
                    id="req-vlm",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=2,
                )
            ]
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].finish_reason == "stop"
        assert events == [
            ("enter", "test-model", None, None),
            ("exit", "test-model", None, None),
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

    def test_vlm_text_model_is_mlx_lm_compatible_with_real_batch_step(self):
        """Batch generation should accept the wrapped VLM language tower."""
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

        from mlx_lm.generate import BatchGenerator

        batch_gen = BatchGenerator(text_model, max_tokens=1, stop_tokens={0})
        sampled, logprobs = batch_gen._step(
            mx.array([[1]]),
            prompt_cache=[],
            samplers=[None],
            logits_processors=[[]],
            tokens=[mx.array([], dtype=mx.int32)],
        )

        assert sampled.tolist() == [1]
        assert len(logprobs) == 1
        assert logprobs[0].shape == (4,)

    def test_vlm_text_model_normalizes_batch_cache_offsets_for_real_insert(self):
        """Real batch prefill should not expose vector offsets to VLM attention."""
        tower = _FakeBatchOffsetSensitiveTower()
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

        from mlx_lm.generate import BatchGenerator

        batch_gen = BatchGenerator(text_model, max_tokens=1, stop_tokens={0})
        uids = batch_gen.insert(
            [[1, 2, 3], [1]],
            max_tokens=[1, 1],
            samplers=[None, None],
        )
        responses = batch_gen.next()

        assert uids == [0, 1]
        assert len(responses) == 2
        assert [response.token for response in responses] == [1, 1]
        assert tower.seen_offset_types
        assert set(tower.seen_offset_types) == {int}
