"""
Tests for prompt cache functionality

This test file verifies the prompt caching functionality in the chat completion API, including:
1. First conversation with no cache
2. Second conversation using cache
3. Modified conversation still hitting partial cache
"""

import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from mlx_batch_server.chat.openai.openai_adapter import OpenAIAdapter
from mlx_batch_server.chat.openai.schema import ChatCompletionRequest, ChatMessage, Role
from mlx_batch_server.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.model


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def openai_client(client):
    """Create OpenAI client configured with test server"""
    return OpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=client,
    )


class TestPromptCache:
    """Tests for prompt cache functionality"""

    def test_openai_adapter_accepts_explicit_null_extra_body(self):
        adapter = OpenAIAdapter(wrapper=object())  # type: ignore[arg-type]
        request = ChatCompletionRequest(
            model="demo-model",
            messages=[ChatMessage(role=Role.USER, content="hello")],
            extra_body=None,
        )

        params = adapter._prepare_generation_params(request)

        assert params["sampler"]["top_k"] == 0

    def test_openai_adapter_honors_prompt_cache_override(self):
        adapter = OpenAIAdapter(wrapper=object())  # type: ignore[arg-type]
        request = ChatCompletionRequest(
            model="demo-model",
            messages=[ChatMessage(role=Role.USER, content="hello")],
            extra_body={"enable_prompt_cache": False},
        )

        params = adapter._prepare_generation_params(request)

        assert params["template_kwargs"]["enable_prompt_cache"] is False
        assert params["enable_prompt_cache"] is False

    def test_openai_adapter_preserves_stream_reasoning_channel(self):
        class FakeWrapper:
            def generate_stream(self, **params):
                assert params["messages"][0]["content"] == "hello"
                yield SimpleNamespace(
                    content=SimpleNamespace(
                        text_delta=None,
                        reasoning_delta="thinking once",
                    ),
                    logprobs=None,
                )
                yield SimpleNamespace(
                    content=SimpleNamespace(
                        text_delta="FINAL",
                        reasoning_delta=None,
                    ),
                    logprobs=None,
                )

        adapter = OpenAIAdapter(wrapper=FakeWrapper())  # type: ignore[arg-type]
        request = ChatCompletionRequest(
            model="demo-model",
            messages=[ChatMessage(role=Role.USER, content="hello")],
            stream=True,
        )

        chunks = list(adapter.generate_stream(request))

        assert chunks[0].choices[0].delta.reasoning == "thinking once"
        assert chunks[0].choices[0].delta.content is None
        assert chunks[1].choices[0].delta.content == "FINAL"
        assert chunks[1].choices[0].delta.reasoning is None

    def test_conversation_with_prompt_cache(self, openai_client):
        try:
            logger.info("\n===== Conversation with prompt cache =====")
            model = "mlx-community/gemma-3-1b-it-4bit-DWQ"
            prompt = "Can you tell me more about your capabilities?"

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]

            first_response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": first_response.choices[0].message.content,
                }
            )
            messages.append({"role": "user", "content": "continue"})

            # Create second conversation
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
            )

            # Verify cache in second conversation
            assert response.usage.prompt_tokens_details is not None, (
                "Second conversation should have cached tokens"
            )
            assert response.usage.prompt_tokens_details.cached_tokens > 0, (
                "Cached tokens count should be greater than 0"
            )
            logger.info(
                f"Second conversation cached tokens: {response.usage.prompt_tokens_details.cached_tokens}"
            )

        except Exception as e:
            logger.error(f"Error testing prompt cache: {e!s}")
            raise
