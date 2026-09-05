import logging

import anthropic
import pytest
from fastapi.testclient import TestClient

from mlx_batch_server.chat.anthropic.anthropic_schema import MessagesRequest
from mlx_batch_server.chat.anthropic.turn_source import (
    AnthropicTurn,
    clear_turn_source,
    register_turn_source,
)
from mlx_batch_server.main import app
from mlx_batch_server.runtime.events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartStarted,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    TurnCompleted,
    TurnStarted,
    UsageUpdate,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _SDKConformanceTurnSource:
    """Deterministic typed owner used to prove the official SDK wire contract."""

    def stream(self, turn: AnthropicTurn):
        async def events():
            yield TurnStarted(
                response_id="anthropic_sdk_conformance",
                model=turn.model_alias,
                created_at=1,
            )
            output_index = 0
            if turn.reasoning.get("enabled") is True:
                yield ContentPartStarted(
                    kind=REASONING_CONTENT_KIND,
                    output_index=output_index,
                    content_index=0,
                    item_id="reasoning_0",
                )
                yield ReasoningDelta(
                    delta="A short deterministic reasoning trace.",
                    item_id="reasoning_0",
                    output_index=output_index,
                    content_index=0,
                )
                yield ReasoningCompleted(
                    text="A short deterministic reasoning trace.",
                    item_id="reasoning_0",
                    output_index=output_index,
                    content_index=0,
                )
                output_index += 1
            yield ContentPartStarted(
                kind=TEXT_CONTENT_KIND,
                output_index=output_index,
                content_index=0,
                item_id="text_0",
            )
            yield TextDelta(
                delta="Hello from the canonical runtime.",
                item_id="text_0",
                output_index=output_index,
                content_index=0,
            )
            yield TextCompleted(
                text="Hello from the canonical runtime.",
                item_id="text_0",
                output_index=output_index,
                content_index=0,
            )
            yield TurnCompleted(
                finish_reason="stop",
                usage=UsageUpdate(input_tokens=7, output_tokens=6, total_tokens=13),
            )

        return events()


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def anthropic_client(client):
    """Create an official Anthropic SDK client pointed at the test server.

    These are conformance tests: they assert that the real SDK parses what
    this server emits through a deterministic typed runtime owner. They do not
    load model weights; live-model parity belongs to the I1 runtime proof.
    """
    source = _SDKConformanceTurnSource()
    register_turn_source(source)
    try:
        yield anthropic.Anthropic(
            base_url="http://test/anthropic",
            api_key="not-needed",
            http_client=client,
        )
    finally:
        clear_turn_source(source)


@pytest.fixture
def direct_client(client):
    """Direct HTTP client for testing raw API responses"""
    return client


class TestAnthropicMessages:
    """Test suite for Anthropic Messages API"""

    thinking_model = "Qwen/Qwen3-0.6B-MLX-4bit"
    model_id = "mlx-community/gemma-3-1b-it-4bit-DWQ"
    max_tokens = 4096

    def test_messages_basic(self, anthropic_client):
        """Test basic message completion"""
        try:
            response = anthropic_client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": "Hello!"}],
            )
            logger.info(f"Anthropic Messages Response:\\n{response}\\n")

            # Validate response structure
            assert response.model == self.model_id, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.type == "message", "Response type is not 'message'"
            assert response.role == "assistant", "Response role is not 'assistant'"
            assert len(response.content) > 0, "No content blocks in response"
            assert response.content[0].type == "text", "First content block is not text"
            assert response.stop_reason is not None, "No stop reason in response"

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise

    def test_messages_basic_text_block(self, anthropic_client):
        """Test basic message completion"""
        try:
            response = anthropic_client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                # messages=[{"role": "user", "content": "Hello!"}],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Why is the ocean salty?"}
                        ],
                    }
                ],
            )
            logger.info(f"Anthropic Messages Response:\\n{response}\\n")

            # Validate response structure
            assert response.model == self.model_id, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert response.type == "message", "Response type is not 'message'"
            assert response.role == "assistant", "Response role is not 'assistant'"
            assert len(response.content) > 0, "No content blocks in response"
            assert response.content[0].type == "text", "First content block is not text"
            assert response.stop_reason is not None, "No stop reason in response"

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise

    def test_messages_conversation(self, anthropic_client):
        """Test multi-turn conversation"""
        try:
            model = "mlx-community/gemma-3-1b-it-4bit-DWQ"
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "Hi there!"},
                    {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    {"role": "user", "content": "Can you explain what AI is?"},
                ],
            )
            logger.info(f"Conversation Response:\\n{response}\\n")
            logger.info(f"Conversation Usage:\\n{response.usage}\\n")

            # Validate response
            assert response.model == model, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert len(response.content) > 0, "No content blocks in response"
            assert response.content[0].type == "text", "First content block is not text"

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise

    def test_messages_with_system_prompt(self, anthropic_client):
        """Test message completion with system prompt"""
        try:
            response = anthropic_client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system="You are a helpful assistant that responds in a friendly manner.",
                messages=[{"role": "user", "content": "What is 2+2?"}],
            )
            logger.info(f"System Prompt Response:\\n{response}\\n")

            # Validate response
            assert response.model == self.model_id, "Model name is not correct"
            assert response.usage is not None, "No usage in response"
            assert len(response.content) > 0, "No content blocks in response"
            assert response.content[0].type == "text", "First content block is not text"

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise

    def test_messages_stream(self, anthropic_client):
        """Test streaming message completion using anthropic_client"""
        try:
            model = "mlx-community/gemma-3-1b-it-4bit-DWQ"

            # Validate streaming response
            event_count = 0
            content_text = ""
            message_start_received = False
            message_delta_received = False
            message_stop_received = False
            content_block_start_received = False
            content_block_stop_received = False
            text_deltas_received = 0

            with anthropic_client.messages.stream(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": "Count from 1 to 5."}],
            ) as stream:
                for event in stream:
                    event_count += 1

                    if event.type == "message_start":
                        message_start_received = True
                        assert event.message.model == model, (
                            "Incorrect model name in stream"
                        )

                    elif event.type == "content_block_start":
                        content_block_start_received = True
                        assert event.content_block is not None, (
                            "Missing content_block in content_block_start"
                        )
                        assert event.index is not None, (
                            "Missing index in content_block_start"
                        )

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # Log all delta details for debugging
                        logger.info(f"Delta details: type={delta.type}, delta={delta}")
                        if delta.type == "text_delta" and delta.text:
                            content_text += delta.text
                            text_deltas_received += 1
                        elif delta.type == "thinking_delta" and hasattr(
                            delta, "thinking"
                        ):
                            # This might happen if model unexpectedly switches to thinking mode
                            logger.warning(
                                f"Received unexpected thinking_delta: {delta.thinking[:50]}..."
                            )
                        elif delta.type:
                            logger.warning(
                                f"Received unexpected delta type: {delta.type}"
                            )
                        else:
                            logger.error(f"Delta missing type field: {delta}")

                    elif event.type == "content_block_stop":
                        content_block_stop_received = True
                        assert event.index is not None, (
                            "Missing index in content_block_stop"
                        )

                    elif event.type == "message_delta":
                        message_delta_received = True
                        delta = event.delta
                        assert delta.stop_reason is not None, (
                            "No stop reason in message_delta"
                        )
                        # Usage should be available in the event
                        assert event.usage is not None, "No usage in message_delta"

                    elif event.type == "message_stop":
                        message_stop_received = True

            # Validate overall streaming response
            assert event_count > 0, "No stream events received"
            assert message_start_received, "No message_start event received"
            assert content_block_start_received, "No content_block_start event received"
            assert content_block_stop_received, "No content_block_stop event received"
            assert message_delta_received, "No message_delta event received"
            assert message_stop_received, "No message_stop event received"
            assert text_deltas_received > 0, (
                f"No text deltas received (got {event_count} total events)"
            )
            assert content_text.strip(), "Generated content is empty"
            logger.info(f"Complete generated content: {content_text}")
            logger.info(f"Received {text_deltas_received} text delta events")

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise

    def test_messages_thinking_stream(self, anthropic_client):
        """Streaming: enabled thinking is refused before the stream opens.

        ``budget_tokens`` has no semantic owner on this runtime. Opening a
        200 SSE stream and quietly ignoring the budget would tell the client
        a reasoning tier was honoured, so W3-AA refuses the request instead
        and W3-AB owns admitting a truthful tier.
        """

        with (
            pytest.raises(anthropic.BadRequestError) as failure,
            anthropic_client.messages.stream(
                model=self.thinking_model,
                max_tokens=self.max_tokens,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[
                    {
                        "role": "user",
                        "content": "Solve this step by step: What is 15 + 27?",
                    }
                ],
            ) as stream,
        ):
            for _event in stream:
                pass

        body = failure.value.response.json()
        assert failure.value.status_code == 400
        assert body["error"]["type"] == "invalid_request_error"
        assert "thinking.type" in body["error"]["message"]
        assert "W3-AB" in body["error"]["message"]

    def test_messages_stream_event_order(self, anthropic_client):
        """验证流式事件的严格顺序：message_start → content_block_start → deltas → content_block_stop → message_delta → message_stop"""
        try:
            model = "mlx-community/gemma-3-1b-it-4bit-DWQ"

            # 严格验证事件顺序
            events_order = []

            with anthropic_client.messages.stream(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": "Say hello"}],
            ) as stream:
                for event in stream:
                    event_type = event.type
                    if event_type:
                        events_order.append(event_type)

            # 验证事件顺序
            logger.info(f"Received events in order: {events_order}")

            # 检查必需的事件都存在
            assert len(events_order) > 0, "No events received at all"
            assert "message_start" in events_order, (
                f"Missing message_start event in {events_order}"
            )
            assert "content_block_start" in events_order, (
                f"Missing content_block_start event in {events_order}"
            )
            assert "content_block_stop" in events_order, (
                f"Missing content_block_stop event in {events_order}"
            )
            assert "message_delta" in events_order, (
                f"Missing message_delta event in {events_order}"
            )
            assert "message_stop" in events_order, (
                f"Missing message_stop event in {events_order}"
            )
            assert events_order.count("content_block_delta") > 0, (
                f"Missing content_block_delta events in {events_order}"
            )

            # 验证严格顺序
            message_start_idx = events_order.index("message_start")
            content_block_start_idx = events_order.index("content_block_start")
            first_delta_idx = events_order.index("content_block_delta")
            content_block_stop_idx = events_order.index("content_block_stop")
            message_delta_idx = events_order.index("message_delta")
            message_stop_idx = events_order.index("message_stop")

            assert message_start_idx < content_block_start_idx, (
                "message_start should come before content_block_start"
            )
            assert content_block_start_idx < first_delta_idx, (
                "content_block_start should come before first content_block_delta"
            )
            assert first_delta_idx < content_block_stop_idx, (
                "content_block_delta should come before content_block_stop"
            )
            assert content_block_stop_idx < message_delta_idx, (
                "content_block_stop should come before message_delta"
            )
            assert message_delta_idx < message_stop_idx, (
                "message_delta should come before message_stop"
            )

            # 验证所有content_block_delta都在start和stop之间
            for i, event in enumerate(events_order):
                if event == "content_block_delta":
                    assert i > content_block_start_idx, (
                        f"content_block_delta at index {i} should come after content_block_start"
                    )
                    assert i < content_block_stop_idx, (
                        f"content_block_delta at index {i} should come before content_block_stop"
                    )

            # 验证message_start是第一个事件，message_stop是最后一个事件
            assert events_order[0] == "message_start", (
                f"message_start should be the first event, got: {events_order[0]}"
            )
            assert events_order[-1] == "message_stop", (
                f"message_stop should be the last event, got: {events_order[-1]}"
            )

            logger.info("✅ Event order validation passed")

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise

    def test_messages_stream_thinking_then_text(self, anthropic_client):
        """The same refusal reaches the streaming transport as an HTTP error.

        The SDK raises before any event is yielded, which is the proof that
        no SSE byte was written for a request the runtime cannot honour.
        """

        with (
            pytest.raises(anthropic.BadRequestError),
            anthropic_client.messages.stream(
                model=self.thinking_model,
                max_tokens=self.max_tokens,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[{"role": "user", "content": "Explain your reasoning."}],
            ) as stream,
        ):
            for _event in stream:
                pytest.fail("a refused request must not emit stream events")

        # Disabled thinking is honoured, so the surface is narrowed, not shut.
        with anthropic_client.messages.stream(
            model=self.thinking_model,
            max_tokens=self.max_tokens,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": "Explain your reasoning."}],
        ) as stream:
            types = [event.type for event in stream]

        assert types[0] == "message_start"
        assert "message_stop" in types
        assert "error" not in types

    def test_messages_thinking_mode(self, anthropic_client):
        """Non-streaming: the identical classification and error body."""

        with pytest.raises(anthropic.BadRequestError) as failure:
            anthropic_client.messages.create(
                model=self.thinking_model,
                max_tokens=self.max_tokens,
                thinking={"type": "enabled", "budget_tokens": 1024},
                messages=[
                    {
                        "role": "user",
                        "content": "Solve this step by step: What is 15 + 27?",
                    }
                ],
            )

        body = failure.value.response.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "thinking.type" in body["error"]["message"]

        # A turn without the unhonoured control still completes.
        response = anthropic_client.messages.create(
            model=self.thinking_model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": "What is 15 + 27?"}],
        )
        assert response.usage is not None
        assert any(block.type == "text" for block in response.content)

    @pytest.mark.parametrize("requested", ["auto", "standard_only"])
    def test_messages_service_tier_reports_the_delivered_lane(
        self, anthropic_client, requested
    ):
        """The official SDK reads the tier that actually served the turn.

        ``auto`` and ``standard_only`` are both accepted here, and neither is
        echoed back: the SDK's own ``usage.service_tier`` — a closed
        ``standard | priority | batch`` set — reports ``standard``, the one
        capacity lane this process runs.
        """

        response = anthropic_client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            service_tier=requested,
            messages=[{"role": "user", "content": "Hello!"}],
        )

        assert response.usage.service_tier == "standard"

    def test_messages_carry_no_thinking_when_none_was_requested(self, anthropic_client):
        """No thinking block appears on a turn that never asked for one."""

        response = anthropic_client.messages.create(
            model=self.thinking_model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": "What is 15 + 27?"}],
        )

        assert all(block.type != "thinking" for block in response.content)
        assert any(block.type == "text" for block in response.content)

    def test_messages_error_handling(self, anthropic_client):
        """Test error handling for invalid requests"""
        try:
            # Test missing required field (max_tokens)
            with pytest.raises(
                Exception
            ):  # anthropic client will raise TypeError for missing required params
                anthropic_client.messages.create(
                    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
                    messages=[{"role": "user", "content": "Hello!"}],
                    # Missing max_tokens
                )

            logger.info("Error handling validation passed")

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise

    def test_messages_schema_validation(self):
        """Test Pydantic schema validation"""
        try:
            # Test valid request
            valid_request = {
                "model": "test-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello!"}],
            }

            request = MessagesRequest(**valid_request)
            assert request.model == "test-model"
            assert request.max_tokens == 100
            assert len(request.messages) == 1

            # Test invalid temperature
            with pytest.raises(ValueError):
                MessagesRequest(
                    model="test-model",
                    max_tokens=self.max_tokens,
                    temperature=2.0,  # Invalid: > 1.0
                    messages=[{"role": "user", "content": "Hello!"}],
                )

            logger.info("Schema validation tests passed")

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise

    def test_usage_tracking(self, anthropic_client):
        """Test that usage statistics are properly tracked"""
        try:
            model = self.model_id
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": "Hi"}],
            )

            # Validate usage statistics
            assert response.usage is not None, "No usage statistics"
            assert response.usage.input_tokens > 0, "No input tokens counted"
            assert response.usage.output_tokens > 0, "No output tokens counted"

            logger.info(
                f"Usage: {response.usage.input_tokens} input, {response.usage.output_tokens} output tokens"
            )

        except Exception as e:
            logger.error(f"Test error: {e!s}")
            raise
