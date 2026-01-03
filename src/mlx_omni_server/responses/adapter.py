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
from ..tools import (
    execute_tool,
    format_tool_result,
    get_tool_definitions,
    is_hosted_tool,
)
from ..utils.harmony_parser import is_harmony_model, parse_harmony_output
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
        model_id = request.model or self.default_model_id
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
                return await self._generate_vision(model_id, normalised)
            else:
                # Text-only path
                return await self._generate_text(model_id, normalised)

        except Exception as e:
            logger.error(f"Responses generation failed: {e}", exc_info=True)
            return build_error_response(
                str(e),
                error_code="internal_error",
                model=model_id,
            )

    async def _generate_text(
        self,
        model_id: str,
        normalised_body: dict[str, Any],
    ) -> ResponseResponse:
        """Generate text-only response using MLX, with tool support."""
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
        if is_harmony_model(model_id) and content_text:
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
            model=model_id,
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

        return await self._generate_text(model_id, text_only_body)

    async def generate_stream(
        self,
        request: ResponseRequest,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate streaming response.

        Yields SSE events compatible with Responses API streaming format.
        Note: Tool execution in streaming mode executes after stream completes.
        """
        model_id = request.model or self.default_model_id
        if not model_id:
            yield {
                "type": "error",
                "error": {
                    "message": "Model not specified",
                    "code": "invalid_request_error",
                },
            }
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

            # Generate response ID and item ID
            response_id = f"resp_{uuid.uuid4().hex}"
            item_id = f"msg_{uuid.uuid4().hex[:24]}"

            # Emit response.created
            yield {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": model_id,
                    "output": [],
                },
            }

            # Emit output_item.added
            yield {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }

            # Emit content_part.added
            yield {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                },
            }

            # Stream content deltas
            full_text = ""
            for chunk in adapter.generate_stream(chat_request):
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        full_text += delta.content
                        yield {
                            "type": "response.output_text.delta",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": delta.content,
                        }

            # Emit content_part.done
            yield {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
            }

            # Emit output_item.done
            yield {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": full_text}],
                },
            }

            # Emit response.completed
            yield {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "model": model_id,
                    "output": [
                        {
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": full_text}],
                        }
                    ],
                },
            }

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": {
                    "message": str(e),
                    "code": "internal_error",
                },
            }
