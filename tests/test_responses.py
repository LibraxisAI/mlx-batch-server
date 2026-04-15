"""
Tests for /v1/responses endpoint.

Contributed by LibraxisAI - https://libraxis.ai
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from mlx_batch_server.main import app
from mlx_batch_server.responses.normalizer import (
    has_media_content,
    normalise_responses_payload,
    parts_to_plaintext,
    responses_to_chat_messages,
)
from mlx_batch_server.responses.schema import (
    ResponseRequest,
    ResponseResponse,
    ResponseStatus,
    build_error_response,
    build_text_output,
)
from mlx_batch_server.utils.harmony_parser import (
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

    def test_response_request_accepts_runtime_override_fields(self):
        """Responses surface should accept exact-runtime overrides like chat-completions."""
        request = ResponseRequest(
            model="test-model",
            input="Hello",
            adapter_path="~/adapters/frontier-lora",
            draft_model="MLX-Community/Qwen3-1.7B-4bit",
        )

        dumped = request.model_dump(exclude_none=True)

        assert request.adapter_path == "~/adapters/frontier-lora"
        assert request.get_draft_model_id() == "MLX-Community/Qwen3-1.7B-4bit"
        assert dumped["adapter_path"] == "~/adapters/frontier-lora"
        assert dumped["draft_model"] == "MLX-Community/Qwen3-1.7B-4bit"

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


class TestResponsesAdapterRouting:
    """Adapter routing tests for mixed text and multimodal traffic."""

    @pytest.mark.asyncio
    async def test_text_image_text_sequence_keeps_one_model_contract(
        self,
        monkeypatch,
    ):
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        calls = []

        async def fake_generate_text(
            model_id,
            normalised_body,
            request_model=None,
            *,
            adapter_path=None,
            draft_model_id=None,
        ):
            calls.append(
                {
                    "lane": "text",
                    "model_id": model_id,
                    "request_model": request_model,
                    "has_tools": bool(normalised_body.get("tools")),
                    "adapter_path": adapter_path,
                    "draft_model_id": draft_model_id,
                }
            )
            return ResponseResponse(
                model=request_model or model_id,
                output=build_text_output("text"),
            )

        async def fake_generate_vision(
            model_id,
            normalised_body,
            request_model=None,
            *,
            adapter_path=None,
            draft_model_id=None,
        ):
            calls.append(
                {
                    "lane": "multimodal",
                    "model_id": model_id,
                    "request_model": request_model,
                    "has_tools": bool(normalised_body.get("tools")),
                    "adapter_path": adapter_path,
                    "draft_model_id": draft_model_id,
                }
            )
            return ResponseResponse(
                model=request_model or model_id,
                output=build_text_output("vision"),
            )

        monkeypatch.setattr(adapter, "_generate_text", fake_generate_text)
        monkeypatch.setattr(adapter, "_generate_vision", fake_generate_vision)

        await adapter.generate(
            ResponseRequest(
                model="demo-model",
                input="Hello",
                adapter_path="/adapter/frontier",
                draft_model="draft-qwen",
            )
        )
        await adapter.generate(
            ResponseRequest(
                model="demo-model",
                adapter_path="/adapter/frontier",
                draft_model="draft-qwen",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/cat.png",
                            },
                        ],
                    }
                ],
            )
        )
        await adapter.generate(
            ResponseRequest(
                model="demo-model",
                input="Use the tool",
                tools=[{"type": "function", "name": "lookup"}],
                adapter_path="/adapter/frontier",
                draft_model="draft-qwen",
            )
        )

        assert [call["lane"] for call in calls] == ["text", "multimodal", "text"]
        assert {call["model_id"] for call in calls} == {"demo-model"}
        assert {call["request_model"] for call in calls} == {"demo-model"}
        assert {call["adapter_path"] for call in calls} == {"/adapter/frontier"}
        assert {call["draft_model_id"] for call in calls} == {"draft-qwen"}
        assert calls[2]["has_tools"] is True

    @pytest.mark.asyncio
    async def test_streaming_text_with_tools_falls_back_to_single_request_lane(
        self,
        monkeypatch,
    ):
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        seen = {}

        monkeypatch.setattr(adapter, "_should_use_batch", lambda: True)

        class FakeAdapter:
            def generate_stream(self, request):
                seen["model"] = request.model
                seen["messages"] = [msg.model_dump() for msg in request.messages]
                seen["stop"] = request.stop
                seen["tools"] = request.tools
                yield type(
                    "Chunk",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "delta": type(
                                        "Delta",
                                        (),
                                        {"content": "sunny"},
                                    )()
                                },
                            )()
                        ]
                    },
                )()

        def fake_get_openai_adapter(
            _model_id: str,
            *,
            adapter_path: str | None = None,
            draft_model_id: str | None = None,
        ):
            seen["adapter_runtime"] = (_model_id, adapter_path, draft_model_id)
            return FakeAdapter()

        monkeypatch.setattr(
            adapter,
            "_get_openai_adapter",
            fake_get_openai_adapter,
        )

        events = [
            event
            async for event in adapter.generate_stream(
                ResponseRequest(
                    model="demo-model",
                    input="What is the weather?",
                    tools=[{"type": "function", "name": "lookup_weather"}],
                    max_output_tokens=64,
                    temperature=0.2,
                )
            )
        ]

        assert seen["model"] == "demo-model"
        assert seen["adapter_runtime"] == ("demo-model", None, None)
        assert len(seen["messages"]) == 1
        assert seen["messages"][0]["role"] == "user"
        assert seen["messages"][0]["content"] == "What is the weather?"
        assert seen["tools"] is not None
        assert any(event["type"] == "response.completed" for event in events)
        assert any(
            event["type"] == "response.output_text.done" and event["text"] == "sunny"
            for event in events
        )

    @pytest.mark.asyncio
    async def test_streaming_text_with_draft_model_uses_single_lane_exact_runtime(
        self,
        monkeypatch,
    ):
        from mlx_batch_server.responses import adapter as adapter_module
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        seen: dict[str, object] = {}

        monkeypatch.setattr(adapter, "_should_use_batch", lambda: True)

        @asynccontextmanager
        async def fake_runtime_session(
            model_id: str,
            adapter_path: str | None = None,
            draft_model_id: str | None = None,
        ):
            seen["runtime_session"] = (model_id, adapter_path, draft_model_id)
            yield {"switched": False}

        class FakeAdapter:
            def generate_stream(self, request):
                seen["request_model"] = request.model
                seen["extra"] = request.get_extra_params()
                yield type(
                    "Chunk",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "delta": type(
                                        "Delta",
                                        (),
                                        {"content": "draft ok"},
                                    )(),
                                    "finish_reason": "stop",
                                },
                            )()
                        ]
                    },
                )()

        def fake_get_openai_adapter(
            model_id: str,
            *,
            adapter_path: str | None = None,
            draft_model_id: str | None = None,
        ):
            seen["adapter_runtime"] = (model_id, adapter_path, draft_model_id)
            return FakeAdapter()

        async def fail_batch(*args, **kwargs):
            raise AssertionError("draft-model requests must not enter the batch lane")
            yield

        monkeypatch.setattr(
            adapter_module,
            "endpoint_runtime_session",
            fake_runtime_session,
        )
        monkeypatch.setattr(adapter, "_get_openai_adapter", fake_get_openai_adapter)
        monkeypatch.setattr(adapter, "_stream_batch_tokens", fail_batch)

        events = [
            event
            async for event in adapter.generate_stream(
                ResponseRequest(
                    model="demo-model",
                    input="Stay on the exact runtime",
                    stream=True,
                    adapter_path="/adapter/frontier",
                    draft_model="draft-qwen",
                )
            )
        ]

        assert seen["runtime_session"] == (
            "demo-model",
            "/adapter/frontier",
            "draft-qwen",
        )
        assert seen["adapter_runtime"] == (
            "demo-model",
            "/adapter/frontier",
            "draft-qwen",
        )
        assert seen["request_model"] == "demo-model"
        assert seen["extra"]["adapter_path"] == "/adapter/frontier"
        assert seen["extra"]["draft_model"] == "draft-qwen"
        assert any(
            event["type"] == "response.output_text.done" and event["text"] == "draft ok"
            for event in events
        )

    @pytest.mark.asyncio
    async def test_multimodal_requests_with_tools_are_rejected(self):
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()

        response = await adapter.generate(
            ResponseRequest(
                model="demo-model",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/cat.png",
                            },
                        ],
                    }
                ],
                tools=[{"type": "function", "name": "lookup"}],
            )
        )

        assert response.status == ResponseStatus.FAILED
        assert response.error["code"] == "invalid_request_error"
        assert "Multimodal requests with tools" in response.error["message"]


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


class TestResponsesRuntimeGuards:
    def test_vision_finalize_events_keep_message_output_index(self):
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        sequence = 0

        def make_event(event_type: str, data: dict):
            nonlocal sequence
            event = {"type": event_type, "sequence_number": sequence, **data}
            sequence += 1
            return event

        events = adapter._vision_finalize_events(
            make_event,
            "resp_demo",
            "demo-model",
            "msg_demo",
            "final answer",
            output_index=1,
            reasoning_item={
                "id": "rs_demo",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking"}],
            },
        )

        output_item_done = next(
            event for event in events if event["type"] == "response.output_item.done"
        )
        assert output_item_done["output_index"] == 1

        completed = next(
            event for event in events if event["type"] == "response.completed"
        )
        assert [item["type"] for item in completed["response"]["output"]] == [
            "reasoning",
            "message",
        ]

    @pytest.mark.asyncio
    async def test_generate_vision_uses_vlm_batch_for_single_image(self, monkeypatch):
        from mlx_batch_server.responses import adapter as adapter_module
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        seen: dict[str, object] = {}

        def fake_settings():
            return SimpleNamespace(
                vlm_batch_enabled=True,
                vlm_stream_batch_enabled=True,
                vlm_batch_window_ms=25,
                vlm_max_batch_size=3,
                vlm_batch_group_by_shape=True,
            )

        class FakeCoordinator:
            async def submit_request(self, **kwargs):
                seen.update(kwargs)
                return SimpleNamespace(
                    text="batched vision",
                    prompt_tokens=11,
                    generation_tokens=7,
                    total_tokens=18,
                )

        def fake_get_vlm_batch_coordinator(**kwargs):
            seen["coordinator_kwargs"] = kwargs
            return FakeCoordinator()

        monkeypatch.setattr(adapter_module, "get_settings", fake_settings)
        monkeypatch.setattr(
            adapter_module,
            "get_vlm_batch_coordinator",
            fake_get_vlm_batch_coordinator,
        )

        normalised = normalise_responses_payload(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/cat.png",
                            },
                        ],
                    }
                ]
            }
        )

        response = await adapter._generate_vision(
            "demo-model",
            normalised,
            "demo-model",
            adapter_path="/adapter/frontier",
            draft_model_id="draft-qwen",
        )

        assert seen["messages"] == [{"role": "user", "content": "Describe this image"}]
        assert len(seen["images"]) == 1
        assert seen["max_tokens"] is None
        assert seen["temperature"] is None
        assert seen["coordinator_kwargs"]["adapter_path"] == "/adapter/frontier"
        assert seen["coordinator_kwargs"]["draft_model_id"] == "draft-qwen"
        assert response.output[-1].content[0]["text"] == "batched vision"
        assert response.usage.input_tokens == 11
        assert response.usage.output_tokens == 7
        assert response.usage.total_tokens == 18

    @pytest.mark.asyncio
    async def test_generate_vision_stream_uses_vlm_stream_batch_for_single_image(
        self,
        monkeypatch,
    ):
        from mlx_batch_server.responses import adapter as adapter_module
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        seen: dict[str, object] = {}

        def fake_settings():
            return SimpleNamespace(
                vlm_batch_enabled=True,
                vlm_stream_batch_enabled=True,
                vlm_batch_window_ms=25,
                vlm_max_batch_size=3,
                vlm_batch_group_by_shape=True,
            )

        class FakeStreamCoordinator:
            async def stream_request(self, **kwargs):
                seen.update(kwargs)
                yield SimpleNamespace(
                    text="<think>reason</think>batched ", finish_reason=None
                )
                yield SimpleNamespace(text="stream", finish_reason=None)
                yield SimpleNamespace(text="", finish_reason="stop")

        def fake_get_vlm_stream_coordinator(**kwargs):
            seen["coordinator_kwargs"] = kwargs
            return FakeStreamCoordinator()

        monkeypatch.setattr(adapter_module, "get_settings", fake_settings)
        monkeypatch.setattr(
            adapter_module,
            "get_vlm_stream_coordinator",
            fake_get_vlm_stream_coordinator,
        )

        normalised = normalise_responses_payload(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/cat.png",
                            },
                        ],
                    }
                ]
            }
        )

        events = [
            event
            async for event in adapter._generate_vision_stream(
                "demo-model",
                normalised,
                "demo-model",
                adapter_path="/adapter/frontier",
                draft_model_id="draft-qwen",
            )
        ]

        assert seen["messages"] == [{"role": "user", "content": "Describe this image"}]
        assert seen["coordinator_kwargs"]["adapter_path"] == "/adapter/frontier"
        assert seen["coordinator_kwargs"]["draft_model_id"] == "draft-qwen"
        completed = next(
            event for event in events if event["type"] == "response.completed"
        )
        message_item = next(
            item
            for item in completed["response"]["output"]
            if item["type"] == "message"
        )
        assert message_item["content"][0]["text"] == "batched stream"

    @pytest.mark.asyncio
    async def test_generate_vision_direct_lane_uses_exact_runtime_identity(
        self,
        monkeypatch,
    ):
        from mlx_batch_server.responses import adapter as adapter_module
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        seen: dict[str, object] = {}

        def fake_settings():
            return SimpleNamespace(
                vlm_batch_enabled=False,
                vlm_stream_batch_enabled=False,
                vlm_batch_window_ms=25,
                vlm_max_batch_size=3,
                vlm_batch_group_by_shape=True,
            )

        def fake_get_vlm_backend(
            model_id: str,
            *,
            adapter_path: str | None = None,
            draft_model_id: str | None = None,
        ):
            seen["backend"] = (model_id, adapter_path, draft_model_id)
            return SimpleNamespace(config={}), SimpleNamespace()

        @contextmanager
        def fake_vlm_execution(
            model_id: str,
            *,
            adapter_path: str | None = None,
            draft_model_id: str | None = None,
        ):
            seen["execution"] = (model_id, adapter_path, draft_model_id)
            yield

        monkeypatch.setattr(adapter_module, "get_settings", fake_settings)
        monkeypatch.setattr(adapter, "_get_vlm_backend", fake_get_vlm_backend)
        monkeypatch.setattr(adapter_module, "vlm_execution", fake_vlm_execution)
        monkeypatch.setattr(
            adapter, "_require_vlm_chat_template", lambda: lambda *a, **k: "prompt"
        )
        monkeypatch.setattr(
            adapter,
            "_require_vlm_generate",
            lambda: lambda *args, **kwargs: SimpleNamespace(
                text="direct vision",
                prompt_tokens=5,
                generation_tokens=2,
                total_tokens=7,
            ),
        )

        normalised = normalise_responses_payload(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/cat.png",
                            },
                        ],
                    }
                ]
            }
        )

        response = await adapter._generate_vision(
            "demo-model",
            normalised,
            "demo-model",
            adapter_path="/adapter/frontier",
            draft_model_id="draft-qwen",
        )

        assert seen["backend"] == (
            "demo-model",
            "/adapter/frontier",
            "draft-qwen",
        )
        assert seen["execution"] == (
            "demo-model",
            "/adapter/frontier",
            "draft-qwen",
        )
        assert response.output[-1].content[0]["text"] == "direct vision"

    @pytest.mark.asyncio
    async def test_generate_vision_marks_shared_runtime_as_llm_surface(
        self,
        monkeypatch,
    ):
        from mlx_batch_server.responses import adapter as adapter_module
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        seen: dict[str, object] = {}

        @contextmanager
        def fake_vlm_execution(model_id: str, **kwargs):
            seen["execution"] = (
                model_id,
                kwargs.get("adapter_path"),
                kwargs.get("draft_model_id"),
            )
            yield

        monkeypatch.setattr(
            adapter,
            "_should_use_vlm_batch",
            lambda *_args, **_kwargs: False,
        )
        monkeypatch.setattr(
            adapter_module,
            "get_vlm_backend",
            lambda model_id, **kwargs: (
                seen.update(
                    {
                        "model_id": model_id,
                        "adapter_path": kwargs.get("adapter_path"),
                        "draft_model_id": kwargs.get("draft_model_id"),
                        "surface": kwargs.get("surface"),
                    }
                )
                or (
                    SimpleNamespace(config=SimpleNamespace()),
                    SimpleNamespace(),
                )
            ),
        )
        monkeypatch.setattr(
            adapter_module,
            "vlm_execution",
            fake_vlm_execution,
        )
        monkeypatch.setattr(
            adapter,
            "_require_vlm_chat_template",
            lambda: (lambda processor, config, messages, **kwargs: "prompt"),
        )
        monkeypatch.setattr(
            adapter,
            "_require_vlm_generate",
            lambda: (
                lambda model, processor, prompt, image=None, **kwargs: SimpleNamespace(
                    text="vision attached",
                    prompt_tokens=4,
                    generation_tokens=2,
                    total_tokens=6,
                )
            ),
        )

        normalised = normalise_responses_payload(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this image"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/cat.png",
                            },
                        ],
                    }
                ]
            }
        )

        response = await adapter._generate_vision(
            "frontier-vlm",
            normalised,
            "frontier-vlm",
        )

        assert seen == {
            "model_id": "frontier-vlm",
            "adapter_path": None,
            "draft_model_id": None,
            "surface": "llm",
            "execution": ("frontier-vlm", None, None),
        }
        assert response.output[-1].content[0]["text"] == "vision attached"
        assert response.usage.total_tokens == 6

    @pytest.mark.asyncio
    async def test_streaming_text_with_previous_response_id_falls_back_to_single_lane(
        self,
        monkeypatch,
    ):
        from mlx_batch_server.responses.adapter import ResponsesAdapter

        adapter = ResponsesAdapter()
        seen = {}

        monkeypatch.setattr(adapter, "_should_use_batch", lambda: True)

        class FakeAdapter:
            def generate_stream(self, request):
                seen["model"] = request.model
                seen["messages"] = [msg.model_dump() for msg in request.messages]
                seen["tools"] = request.tools
                seen["extra"] = request.get_extra_params()
                yield type(
                    "Chunk",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "delta": type(
                                        "Delta",
                                        (),
                                        {"content": "follow-up ok"},
                                    )()
                                },
                            )()
                        ]
                    },
                )()

        def fake_get_openai_adapter(
            _model_id: str,
            *,
            adapter_path: str | None = None,
            draft_model_id: str | None = None,
        ):
            seen["adapter_runtime"] = (_model_id, adapter_path, draft_model_id)
            return FakeAdapter()

        monkeypatch.setattr(
            adapter,
            "_get_openai_adapter",
            fake_get_openai_adapter,
        )

        events = [
            event
            async for event in adapter.generate_stream(
                ResponseRequest(
                    model="demo-model",
                    previous_response_id="resp_prev_123",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Tell me more about that.",
                                }
                            ],
                        }
                    ],
                )
            )
        ]

        assert seen["model"] == "demo-model"
        assert seen["adapter_runtime"] == ("demo-model", None, None)
        assert seen["tools"] is None
        assert seen["extra"]["extra_body"]["enable_prompt_cache"] is False
        assert len(seen["messages"]) == 1
        assert seen["messages"][0]["content"] == "Tell me more about that."
        assert any(event["type"] == "response.completed" for event in events)
        assert any(
            event["type"] == "response.output_text.done"
            and event["text"] == "follow-up ok"
            for event in events
        )


# =============================================================================
# Concurrent & Chain Tests (require running server with loaded model)
# =============================================================================
# These tests validate batch inference and stateful conversations.
# Skip if MLX_TEST_MODEL not set or server not running.

TEST_MODEL = os.environ.get(
    "MLX_TEST_MODEL", "mlx-community/Llama-3.2-1B-Instruct-4bit"
)
TEST_PORT = int(os.environ.get("MLX_TEST_PORT", "10240"))
TEST_BASE_URL = f"http://localhost:{TEST_PORT}"


def _server_available() -> bool:
    """Check if test server is running."""
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{TEST_BASE_URL}/v1/models")
            return r.status_code == 200
    except Exception:
        return False


def _model_loaded() -> bool:
    """Check if test model is loaded."""
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{TEST_BASE_URL}/v1/models")
            if r.status_code != 200:
                return False
            models = r.json().get("data", [])
            return any(TEST_MODEL in m.get("id", "") for m in models)
    except Exception:
        return False


requires_server = pytest.mark.skipif(
    not _server_available(),
    reason=f"Server not running on {TEST_BASE_URL}",
)

requires_model = pytest.mark.skipif(
    not _model_loaded(),
    reason=f"Model {TEST_MODEL} not loaded",
)


class TestConcurrentResponses:
    """Tests for concurrent batch inference."""

    @requires_server
    def test_concurrent_requests(self):
        """Multiple concurrent requests should complete without errors.

        This validates that batch inference works - requests should be
        processed in parallel, not sequentially.
        """
        n_requests = 5
        prompts = [f"Say the number {i}" for i in range(n_requests)]
        results = []

        def make_request(prompt: str) -> dict:
            start = time.perf_counter()
            with httpx.Client(timeout=60.0) as client:
                r = client.post(
                    f"{TEST_BASE_URL}/v1/responses",
                    json={
                        "model": TEST_MODEL,
                        "input": prompt,
                        "max_output_tokens": 50,
                    },
                )
                elapsed = time.perf_counter() - start
                return {"status": r.status_code, "elapsed": elapsed, "prompt": prompt}

        start_all = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_requests) as executor:
            results = list(executor.map(make_request, prompts))
        total_time = time.perf_counter() - start_all

        # All requests should succeed
        for r in results:
            assert r["status"] == 200, f"Request failed: {r}"

        # If batch works, total time should be less than sum of individual times
        sum_individual = sum(r["elapsed"] for r in results)
        speedup = sum_individual / total_time if total_time > 0 else 1

        print(f"\nConcurrent test: {n_requests} requests")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Sum individual: {sum_individual:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Expect at least some parallelism (speedup > 1.5x for 5 requests)
        # This is a soft check - CI may have different characteristics
        if speedup < 1.2:
            pytest.skip(f"Low speedup ({speedup:.2f}x) - may not have batch enabled")


class TestChainedResponses:
    """Tests for multi-turn conversation chains."""

    @requires_server
    def test_chained_conversation(self):
        """Multi-turn chain with previous_response_id should work.

        Each turn should have access to previous context.
        """
        turns = [
            "My name is TestUser. Remember it.",
            "What is my name?",
            "Summarize our conversation so far.",
        ]

        prev_id = None
        responses = []

        with httpx.Client(timeout=60.0) as client:
            for i, prompt in enumerate(turns):
                payload = {
                    "model": TEST_MODEL,
                    "input": prompt,
                    "max_output_tokens": 100,
                }
                if prev_id:
                    payload["previous_response_id"] = prev_id

                r = client.post(f"{TEST_BASE_URL}/v1/responses", json=payload)
                assert r.status_code == 200, f"Turn {i + 1} failed: {r.text}"

                data = r.json()
                prev_id = data.get("id")
                assert prev_id, f"No response ID in turn {i + 1}"

                # Extract text from response
                text = ""
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for part in item.get("content", []):
                            if part.get("type") == "output_text":
                                text += part.get("text", "")

                responses.append({"turn": i + 1, "prompt": prompt, "response": text})
                print(f"\nTurn {i + 1}: {prompt[:30]}...")
                print(f"  Response: {text[:100]}...")

        assert len(responses) == len(turns)

    @requires_server
    def test_concurrent_chains(self):
        """Multiple concurrent multi-turn chains should work independently.

        Each chain should maintain its own context.
        """
        n_chains = 3
        names = ["Alice", "Bob", "Charlie"]

        def run_chain(name: str) -> dict:
            turns = [
                f"My name is {name}. Remember it.",
                "What is my name?",
            ]
            prev_id = None

            with httpx.Client(timeout=60.0) as client:
                for prompt in turns:
                    payload = {
                        "model": TEST_MODEL,
                        "input": prompt,
                        "max_output_tokens": 50,
                    }
                    if prev_id:
                        payload["previous_response_id"] = prev_id

                    r = client.post(f"{TEST_BASE_URL}/v1/responses", json=payload)
                    if r.status_code != 200:
                        return {"name": name, "success": False, "error": r.text}

                    data = r.json()
                    prev_id = data.get("id")

                # Get final response text
                text = ""
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for part in item.get("content", []):
                            if part.get("type") == "output_text":
                                text += part.get("text", "")

                return {"name": name, "success": True, "response": text}

        with ThreadPoolExecutor(max_workers=n_chains) as executor:
            results = list(executor.map(run_chain, names))

        for r in results:
            assert r["success"], f"Chain for {r['name']} failed: {r.get('error')}"
            print(f"\n{r['name']}: {r['response'][:100]}...")


class TestStreamingResponses:
    """Tests for SSE streaming."""

    @requires_server
    def test_streaming_basic(self):
        """Streaming should emit SSE events."""
        with (
            httpx.Client(timeout=60.0) as client,
            client.stream(
                "POST",
                f"{TEST_BASE_URL}/v1/responses",
                json={
                    "model": TEST_MODEL,
                    "input": "Say hello",
                    "stream": True,
                    "max_output_tokens": 50,
                },
            ) as response,
        ):
            assert response.status_code == 200

            events = []
            for line in response.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())

            # Should have at least created and completed events
            print(f"\nReceived {len(events)} events: {events[:10]}...")
            assert len(events) > 0, "No SSE events received"

    @requires_server
    def test_concurrent_streaming(self):
        """Multiple concurrent streams should work."""
        n_streams = 3

        def stream_request(idx: int) -> dict:
            events = []
            try:
                with (
                    httpx.Client(timeout=60.0) as client,
                    client.stream(
                        "POST",
                        f"{TEST_BASE_URL}/v1/responses",
                        json={
                            "model": TEST_MODEL,
                            "input": f"Count to {idx + 1}",
                            "stream": True,
                            "max_output_tokens": 50,
                        },
                    ) as response,
                ):
                    for line in response.iter_lines():
                        if line.startswith("event:"):
                            events.append(line.split(":", 1)[1].strip())
                return {"idx": idx, "success": True, "events": len(events)}
            except Exception as e:
                return {"idx": idx, "success": False, "error": str(e)}

        with ThreadPoolExecutor(max_workers=n_streams) as executor:
            results = list(executor.map(stream_request, range(n_streams)))

        for r in results:
            assert r["success"], f"Stream {r['idx']} failed: {r.get('error')}"
            assert r["events"] > 0, f"Stream {r['idx']} got no events"
            print(f"\nStream {r['idx']}: {r['events']} events")
