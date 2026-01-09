"""
Responses API adapter - bridges to chat completions.

Handles conversion between Responses API format and chat completions,
with support for both local MLX models and external providers.

Includes hosted tool execution (web_search, code_interpreter).

Created by M&K (c)2026 The LibraxisAI Team
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from ..chat.mlx.wrapper_cache import wrapper_cache
from ..chat.openai.openai_adapter import OpenAIAdapter
from ..chat.openai.schema import ChatCompletionRequest, ChatMessage, Role, Tool
from ..core.config import get_settings
from ..tools import (
    execute_tool,
    format_tool_result,
    get_tool_definitions,
    is_hosted_tool,
)
from ..utils.harmony_parser import (
    HarmonyStreamingParser,
    is_harmony_model,
    parse_harmony_output,
)
from ..utils.logger import logger
from .normalizer import (
    has_media_content,
    normalise_responses_payload,
    responses_to_chat_messages,
)
from .schema import (
    ResponseRequest,
    ResponseResponse,
    ResponseStatus,
    ResponseUsage,
    build_error_response,
    build_text_output,
)

if TYPE_CHECKING:
    from ..chat.mlx.chat_generator import ChatGenerator


class ResponsesAdapter:
    """
    Adapter for Responses API that routes to appropriate backend.

    Supports:
    - Local MLX models via ChatGenerator
    - External providers via multi-provider routing
    - Vision models via Ollama routing
    - Hosted tool execution (web_search, code_interpreter)
    """

    def __init__(self, model_id: str | None = None):
        """
        Initialize adapter.

        Args:
            model_id: Default model to use (can be overridden per request)
        """
        self.default_model_id = model_id

    def _get_chat_generator(self, model_id: str) -> ChatGenerator:
        """Get or create ChatGenerator for model (uses shared cache)."""
        return wrapper_cache.get_wrapper(model_id, None, None)

    def _get_openai_adapter(self, model_id: str) -> OpenAIAdapter:
        """Get OpenAI adapter wrapping ChatGenerator."""
        generator = self._get_chat_generator(model_id)
        return OpenAIAdapter(generator)

    async def generate(
        self,
        request: ResponseRequest,
    ) -> ResponseResponse:
        """
        Generate response for Responses API request.

        Routes based on:
        - Media content → vision model
        - Text-only → primary LLM

        Supports hosted tools (web_search, code_interpreter).

        Args:
            request: ResponseRequest body

        Returns:
            ResponseResponse with generated content
        """
        # Preserve original model name for response (don't expose local paths)
        request_model = request.model or self.default_model_id
        model_id = request_model

        if not model_id:
            return build_error_response(
                "Model not specified",
                error_code="invalid_request_error",
            )

        try:
            # Normalize request
            body = request.model_dump(exclude_none=True)
            normalised = normalise_responses_payload(body)

            # Check for media content
            if has_media_content(normalised):
                # Route to vision model
                return await self._generate_vision(model_id, normalised, request_model)
            else:
                # Text-only path
                return await self._generate_text(model_id, normalised, request_model)

        except Exception as e:
            logger.error(f"Responses generation failed: {e}", exc_info=True)
            return build_error_response(
                str(e),
                error_code="internal_error",
                model=request_model,
            )

    async def _generate_text(
        self,
        model_id: str,
        normalised_body: dict[str, Any],
        request_model: str | None = None,
    ) -> ResponseResponse:
        """Generate text-only response using MLX, with tool support."""
        # Use request_model for response, fallback to model_id
        response_model = request_model or model_id
        # Convert to chat messages
        messages = responses_to_chat_messages(normalised_body)

        # Expand hosted tools to function definitions
        request_tools = normalised_body.get("tools")
        tool_definitions = get_tool_definitions(request_tools)

        # Convert to Tool objects if we have tools
        tools_for_request = None
        if tool_definitions:
            tools_for_request = [
                Tool(
                    type="function",
                    function={
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                )
                for t in tool_definitions
                if t.get("type") == "function"
            ]

        # Build ChatCompletionRequest
        chat_request = ChatCompletionRequest(
            model=model_id,
            messages=[
                ChatMessage(role=Role(msg["role"]), content=msg["content"])
                for msg in messages
            ],
            max_tokens=normalised_body.get("max_output_tokens")
            or normalised_body.get("max_tokens"),
            temperature=normalised_body.get("temperature"),
            top_p=normalised_body.get("top_p"),
            stop=normalised_body.get("stop"),
            stream=False,
            tools=tools_for_request,
        )

        # Get adapter and generate
        adapter = self._get_openai_adapter(model_id)
        completion = adapter.generate(chat_request)

        # Extract content from completion
        choice = completion.choices[0] if completion.choices else None
        content_text = ""
        reasoning_text = None
        tool_calls = []

        if choice and choice.message:
            content_text = choice.message.content or ""
            reasoning_text = getattr(choice.message, "reasoning", None)
            tool_calls = getattr(choice.message, "tool_calls", None) or []

        # Parse Harmony format if applicable (GPT-OSS models)
        # Resolve model alias to check Harmony (e.g., "chat" -> "gpt-oss-120b")
        settings = get_settings()
        resolved_model = settings.get_model_alias(model_id)
        if (
            is_harmony_model(resolved_model) or is_harmony_model(model_id)
        ) and content_text:
            parsed = parse_harmony_output(content_text)
            content_text = parsed["final_text"]
            reasoning_text = parsed["reasoning"] or reasoning_text

            # Extract tool calls from Harmony format
            if parsed["tool_calls"] and not tool_calls:
                tool_calls = [
                    type(
                        "ToolCall",
                        (),
                        {
                            "id": tc["id"],
                            "function": type(
                                "Function",
                                (),
                                {"name": tc["name"], "arguments": tc["arguments"]},
                            )(),
                        },
                    )()
                    for tc in parsed["tool_calls"]
                ]
                logger.info(f"Parsed {len(tool_calls)} tool calls from Harmony format")

        # Build output items
        output_items = build_text_output(content_text, reasoning_text)

        # Process tool calls if any
        if tool_calls:
            for tc in tool_calls:
                call_id = (
                    tc.id if hasattr(tc, "id") else f"call_{uuid.uuid4().hex[:24]}"
                )
                func = tc.function if hasattr(tc, "function") else tc
                tool_name = func.name if hasattr(func, "name") else func.get("name", "")
                args_str = (
                    func.arguments
                    if hasattr(func, "arguments")
                    else func.get("arguments", "{}")
                )

                # Add function_call to output
                output_items.append(
                    {
                        "id": call_id,
                        "type": "function_call",
                        "name": tool_name,
                        "arguments": args_str,
                        "call_id": call_id,
                        "status": "completed",
                    }
                )

                # Execute hosted tools
                if is_hosted_tool(tool_name):
                    try:
                        args = (
                            json.loads(args_str)
                            if isinstance(args_str, str)
                            else args_str
                        )
                        result = await execute_tool(tool_name, args)
                        formatted = format_tool_result(tool_name, call_id, result)
                        output_items.append(formatted)
                        logger.info(f"Executed hosted tool: {tool_name}")
                    except Exception as e:
                        logger.error(f"Tool execution failed for {tool_name}: {e}")
                        output_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": f"Error: {e}",
                            }
                        )

        return ResponseResponse(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=int(time.time()),
            model=response_model,
            status=ResponseStatus.COMPLETED,
            output=output_items,
            usage=ResponseUsage(
                input_tokens=completion.usage.prompt_tokens if completion.usage else 0,
                output_tokens=completion.usage.completion_tokens
                if completion.usage
                else 0,
                total_tokens=completion.usage.total_tokens if completion.usage else 0,
            ),
            _provider="mlx-omni-server",
        )

    async def _generate_vision(
        self,
        model_id: str,
        normalised_body: dict[str, Any],
        request_model: str | None = None,
    ) -> ResponseResponse:
        """
        Generate response for vision/multimodal content.

        Currently routes to Ollama for vision models.
        Can be extended to support other vision providers.
        """
        # For now, fall back to text extraction
        # Vision routing will be added in future iteration
        logger.warning(
            f"Vision content detected but vision routing not yet implemented. "
            f"Falling back to text-only processing for model {model_id}"
        )

        # Extract text from multimodal content
        text_only_body = dict(normalised_body)
        for turn in text_only_body.get("input", []):
            if isinstance(turn, dict):
                turn["content"] = [
                    p
                    for p in turn.get("content", [])
                    if isinstance(p, dict) and p.get("type") == "input_text"
                ]

        return await self._generate_text(model_id, text_only_body, request_model)

    async def generate_stream(  # noqa: PLR0912, PLR0915
        self,
        request: ResponseRequest,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate streaming response with full OpenAI Responses API compliance.

        Yields SSE events compatible with official OpenAI format including:
        - sequence_number in every event
        - response.in_progress event
        - response.reasoning_summary_text.delta for Harmony analysis channel
        - response.output_text.delta for final content

        Note: Tool execution in streaming mode executes after stream completes.
        """
        # Preserve original model name for response (don't expose local paths)
        request_model = request.model or self.default_model_id
        model_id = request_model  # May be resolved to full path internally

        # Sequence number counter for OpenAI compliance
        seq_num = 0

        def make_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
            """Create event with sequence_number."""
            nonlocal seq_num
            event = {"type": event_type, "sequence_number": seq_num, **data}
            seq_num += 1
            return event

        if not model_id:
            yield make_event(
                "error",
                {
                    "error": {
                        "message": "Model not specified",
                        "code": "invalid_request_error",
                    }
                },
            )
            return

        try:
            # Normalize request
            body = request.model_dump(exclude_none=True)
            normalised = normalise_responses_payload(body)

            # Convert to chat messages
            messages = responses_to_chat_messages(normalised)

            # Expand hosted tools
            request_tools = normalised.get("tools")
            tool_definitions = get_tool_definitions(request_tools)

            tools_for_request = None
            if tool_definitions:
                tools_for_request = [
                    Tool(
                        type="function",
                        function={
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        },
                    )
                    for t in tool_definitions
                    if t.get("type") == "function"
                ]

            # Build streaming request
            chat_request = ChatCompletionRequest(
                model=model_id,
                messages=[
                    ChatMessage(role=Role(msg["role"]), content=msg["content"])
                    for msg in messages
                ],
                max_tokens=normalised.get("max_output_tokens")
                or normalised.get("max_tokens"),
                temperature=normalised.get("temperature"),
                top_p=normalised.get("top_p"),
                stop=normalised.get("stop"),
                stream=True,
                tools=tools_for_request,
            )

            # Get adapter
            adapter = self._get_openai_adapter(model_id)

            # Generate IDs
            response_id = f"resp_{uuid.uuid4().hex[:6]}"
            reasoning_item_id = f"rs_{uuid.uuid4().hex[:6]}"
            message_item_id = f"msg_{uuid.uuid4().hex[:6]}"

            # Resolve model alias to check Harmony format (e.g., "chat" -> "gpt-oss-120b")
            settings = get_settings()
            resolved_model = settings.get_model_alias(model_id)
            is_harmony = is_harmony_model(resolved_model) or is_harmony_model(model_id)

            # Response object template
            response_obj = {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": request_model,
                "output": [],
            }

            # Emit response.created
            yield make_event(
                "response.created",
                {"response": response_obj.copy()},
            )

            # Emit response.in_progress (OpenAI compliance)
            yield make_event(
                "response.in_progress",
                {"response": response_obj.copy()},
            )

            # For Harmony models: emit reasoning output_item first
            if is_harmony:
                yield make_event(
                    "response.output_item.added",
                    {
                        "output_index": 0,
                        "item": {
                            "id": reasoning_item_id,
                            "type": "reasoning",
                            "summary": [],
                        },
                    },
                )

            # Initialize streaming state
            harmony_parser = HarmonyStreamingParser() if is_harmony else None
            reasoning_text_parts: list[str] = []
            output_text_parts: list[str] = []
            message_item_emitted = False

            # Stream content deltas
            for chunk in adapter.generate_stream(chat_request):
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if not delta or not delta.content:
                    continue

                raw_content = delta.content

                if is_harmony and harmony_parser:
                    # Use stateful parser for channel separation
                    event_type, clean_text = harmony_parser.process_delta(raw_content)

                    if not clean_text:
                        continue  # Skip marker-only chunks

                    if event_type == "reasoning":
                        # Emit reasoning delta
                        reasoning_text_parts.append(clean_text)
                        yield make_event(
                            "response.reasoning_summary_text.delta",
                            {
                                "item_id": reasoning_item_id,
                                "output_index": 0,
                                "delta": clean_text,
                            },
                        )
                    elif event_type == "output":
                        # First output chunk: emit message item
                        if not message_item_emitted:
                            # Close reasoning item if we had one
                            if reasoning_text_parts:
                                yield make_event(
                                    "response.reasoning_summary_text.done",
                                    {
                                        "item_id": reasoning_item_id,
                                        "output_index": 0,
                                        "text": "".join(reasoning_text_parts),
                                    },
                                )
                                yield make_event(
                                    "response.output_item.done",
                                    {
                                        "output_index": 0,
                                        "item": {
                                            "id": reasoning_item_id,
                                            "type": "reasoning",
                                            "summary": [
                                                {
                                                    "type": "summary_text",
                                                    "text": "".join(
                                                        reasoning_text_parts
                                                    ),
                                                }
                                            ],
                                        },
                                    },
                                )

                            # Emit message output_item.added
                            yield make_event(
                                "response.output_item.added",
                                {
                                    "output_index": 1 if reasoning_text_parts else 0,
                                    "item": {
                                        "id": message_item_id,
                                        "type": "message",
                                        "role": "assistant",
                                        "status": "in_progress",
                                        "content": [],
                                    },
                                },
                            )
                            yield make_event(
                                "response.content_part.added",
                                {
                                    "output_index": 1 if reasoning_text_parts else 0,
                                    "content_index": 0,
                                    "part": {"type": "output_text", "text": ""},
                                },
                            )
                            message_item_emitted = True

                        # Emit output delta
                        output_text_parts.append(clean_text)
                        yield make_event(
                            "response.output_text.delta",
                            {
                                "output_index": 1 if reasoning_text_parts else 0,
                                "content_index": 0,
                                "delta": clean_text,
                            },
                        )
                else:
                    # Non-Harmony model: simple streaming
                    if not message_item_emitted:
                        yield make_event(
                            "response.output_item.added",
                            {
                                "output_index": 0,
                                "item": {
                                    "id": message_item_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                            },
                        )
                        yield make_event(
                            "response.content_part.added",
                            {
                                "output_index": 0,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": ""},
                            },
                        )
                        message_item_emitted = True

                    output_text_parts.append(raw_content)
                    yield make_event(
                        "response.output_text.delta",
                        {
                            "output_index": 0,
                            "content_index": 0,
                            "delta": raw_content,
                        },
                    )

            # Parse final text for Harmony models
            final_text = "".join(output_text_parts)
            reasoning_text = (
                "".join(reasoning_text_parts) if reasoning_text_parts else None
            )

            if is_harmony and harmony_parser:
                # Final parse for clean output and tool calls
                parsed = parse_harmony_output(harmony_parser.full_text)
                if parsed["final_text"]:
                    final_text = parsed["final_text"]
                if parsed["reasoning"] and not reasoning_text:
                    reasoning_text = parsed["reasoning"]
                if parsed["tool_calls"]:
                    logger.info(
                        f"Extracted {len(parsed['tool_calls'])} tool calls from Harmony stream"
                    )

            # Handle case where no message was emitted (all reasoning)
            if not message_item_emitted:
                output_index = 1 if reasoning_text_parts else 0
                yield make_event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": {
                            "id": message_item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    },
                )
                yield make_event(
                    "response.content_part.added",
                    {
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": ""},
                    },
                )

            # Emit final events
            output_index = 1 if reasoning_text_parts else 0

            yield make_event(
                "response.output_text.done",
                {
                    "output_index": output_index,
                    "content_index": 0,
                    "text": final_text,
                },
            )

            yield make_event(
                "response.content_part.done",
                {
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": final_text},
                },
            )

            yield make_event(
                "response.output_item.done",
                {
                    "output_index": output_index,
                    "item": {
                        "id": message_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": final_text}],
                    },
                },
            )

            # Build final output list
            output_list = []
            if reasoning_text_parts:
                output_list.append(
                    {
                        "id": reasoning_item_id,
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": reasoning_text or ""}
                        ],
                    }
                )
            output_list.append(
                {
                    "id": message_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": final_text}],
                }
            )

            # Emit response.completed
            yield make_event(
                "response.completed",
                {
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "model": request_model,
                        "output": output_list,
                    }
                },
            )

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            yield make_event(
                "error",
                {
                    "error": {
                        "message": str(e),
                        "code": "internal_error",
                    }
                },
            )
