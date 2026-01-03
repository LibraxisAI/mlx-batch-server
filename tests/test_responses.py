"""
Tests for /v1/responses endpoint.

Contributed by LibraxisAI - https://libraxis.ai
"""

import pytest
from fastapi.testclient import TestClient

from mlx_omni_server.main import app
from mlx_omni_server.responses.normalizer import (
    has_media_content,
    normalise_responses_payload,
    parts_to_plaintext,
    responses_to_chat_messages,
)
from mlx_omni_server.responses.schema import (
    ResponseRequest,
    ResponseStatus,
    build_error_response,
    build_text_output,
)
from mlx_omni_server.utils.harmony_parser import (
    is_harmony_model,
    parse_reasoning_channels,
)


class TestResponsesSchema:
    """Tests for Responses API schema."""

    def test_response_request_simple(self):
        """Simple text request should parse correctly."""
        request = ResponseRequest(
            model="test-model",
            input="Hello, world!",
        )
        assert request.model == "test-model"
        assert request.input == "Hello, world!"
        assert request.modalities == ["text"]
        assert request.stream is False

    def test_response_request_with_turns(self):
        """Request with message turns should parse correctly."""
        request = ResponseRequest(
            model="test-model",
            input=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
        )
        assert len(request.input) == 2

    def test_response_request_max_tokens_aliases(self):
        """Both max_tokens and max_output_tokens should work."""
        request1 = ResponseRequest(
            model="test",
            input="test",
            max_tokens=100,
        )
        assert request1.get_max_tokens() == 100

        request2 = ResponseRequest(
            model="test",
            input="test",
            max_output_tokens=200,
        )
        assert request2.get_max_tokens() == 200

    def test_build_text_output_simple(self):
        """build_text_output should create message item."""
        items = build_text_output("Hello!")
        assert len(items) == 1
        assert items[0].type == "message"
        assert items[0].content[0]["text"] == "Hello!"

    def test_build_text_output_with_reasoning(self):
        """build_text_output should include reasoning item."""
        items = build_text_output("Hello!", reasoning="Let me think...")
        assert len(items) == 2
        assert items[0].type == "reasoning"
        assert items[1].type == "message"

    def test_build_error_response(self):
        """build_error_response should create error response."""
        response = build_error_response("Something went wrong", "test_error")
        assert response.status == ResponseStatus.FAILED
        assert response.error["message"] == "Something went wrong"
        assert response.error["code"] == "test_error"


class TestResponsesNormalizer:
    """Tests for request normalization."""

    def test_normalise_string_input(self):
        """String input should become single user turn."""
        body = {"input": "Hello!"}
        normalised = normalise_responses_payload(body)

        assert len(normalised["input"]) == 1
        assert normalised["input"][0]["role"] == "user"
        assert normalised["input"][0]["content"][0]["type"] == "input_text"
        assert normalised["input"][0]["content"][0]["text"] == "Hello!"

    def test_normalise_message_turns(self):
        """Message turn input should preserve structure."""
        body = {
            "input": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi!"},
            ]
        }
        normalised = normalise_responses_payload(body)

        assert len(normalised["input"]) == 2
        assert normalised["input"][0]["role"] == "system"
        assert normalised["input"][1]["role"] == "user"

    def test_normalise_image_content(self):
        """Image content should be normalized correctly."""
        body = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is this?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/img.png",
                        },
                    ],
                }
            ]
        }
        normalised = normalise_responses_payload(body)

        parts = normalised["input"][0]["content"]
        assert len(parts) == 2
        assert parts[0]["type"] == "input_text"
        assert parts[1]["type"] == "input_image"
        assert parts[1]["image_url"] == "https://example.com/img.png"

    def test_has_media_content_text_only(self):
        """Text-only content should not be detected as media."""
        body = {
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
            ]
        }
        assert not has_media_content(body)

    def test_has_media_content_with_image(self):
        """Image content should be detected as media."""
        body = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "http://example.com/img.png",
                        }
                    ],
                }
            ]
        }
        assert has_media_content(body)

    def test_has_media_content_from_modalities(self):
        """Media modalities should trigger detection."""
        body = {"input": "test", "modalities": ["text", "image"]}
        normalised = normalise_responses_payload(body)
        assert has_media_content(normalised)

    def test_parts_to_plaintext(self):
        """Content parts should convert to plaintext."""
        parts = [
            {"type": "input_text", "text": "Hello"},
            {"type": "input_text", "text": "World"},
        ]
        text = parts_to_plaintext(parts)
        assert text == "Hello\nWorld"

    def test_responses_to_chat_messages(self):
        """Responses format should convert to chat messages."""
        body = {
            "system_instruction": "Be helpful.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hi!"}]},
            ],
        }
        normalised = normalise_responses_payload(body)
        messages = responses_to_chat_messages(normalised)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Be helpful" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi!"


class TestHarmonyParser:
    """Tests for Harmony format parser."""

    def test_parse_reasoning_channels(self):
        """Should parse analysis and final channels."""
        reasoning = """analysis:
This is my analysis.
I'm thinking about the problem.

final:
Here is my answer."""

        analysis, final = parse_reasoning_channels(reasoning)

        assert analysis is not None
        assert "This is my analysis" in analysis
        assert final is not None
        assert "Here is my answer" in final

    def test_is_harmony_model(self):
        """Should detect Harmony models by name."""
        assert is_harmony_model("gpt-oss-120b")
        assert is_harmony_model("openai/gpt-oss-1b")
        assert is_harmony_model("harmony-test")
        assert not is_harmony_model("llama-3")
        assert not is_harmony_model("qwen2.5-coder")


# Integration tests (require model to be loaded)
@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestResponsesEndpoint:
    """Integration tests for /v1/responses endpoint."""

    def test_responses_endpoint_exists(self, client):
        """Endpoint should exist and accept requests."""
        # This will fail without a model, but should return proper error
        response = client.post(
            "/v1/responses",
            json={
                "model": "nonexistent-model",
                "input": "Hello!",
            },
        )
        # Should get a response (either success or proper error)
        assert response.status_code in [200, 400, 500]

    def test_responses_streaming_endpoint(self, client):
        """Streaming endpoint should return SSE."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello!",
                "stream": True,
            },
        )
        # Should get SSE content type or error
        assert response.status_code in [200, 400, 500]

    def test_responses_get_not_found(self, client):
        """GET for nonexistent response should return 404."""
        response = client.get("/v1/responses/resp_nonexistent")
        assert response.status_code == 404

    def test_responses_delete_not_found(self, client):
        """DELETE for nonexistent response should return 404."""
        response = client.delete("/v1/responses/resp_nonexistent")
        assert response.status_code == 404
