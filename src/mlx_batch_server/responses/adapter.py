"""
Responses API adapter - bridges to chat completions.

Handles conversion between Responses API format and chat completions,
with support for both local MLX models and external providers.

Includes hosted tool execution (web_search, code_interpreter).

Vibecrafted with AI Agents by VetCoders (c)2026 The LibraxisAI Team
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from PIL import Image

from ..batch import BatchStreamChunk, get_batch_coordinator
from ..chat.mlx.chat_generator import ChatGenerator
from ..chat.mlx.runtime_policy import endpoint_runtime_session
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
    ReasoningStreamingParser,
    is_harmony_model,
    parse_harmony_output,
    parse_reasoning_like_output,
)
from ..utils.logger import logger
from ..utils.video_loader import build_video_prompt_and_inputs
from ..vision.vlm_batch import (
    get_vlm_batch_coordinator,
    get_vlm_stream_coordinator,
)
from .normalizer import (
    collect_system_preamble,
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
    from collections.abc import AsyncGenerator

try:  # Optional dependency for multimodal lanes.
    from mlx_vlm import apply_chat_template as _mlx_vlm_apply_chat_template
except Exception as exc:  # pragma: no cover - optional dependency
    _mlx_vlm_apply_chat_template = None
    _mlx_vlm_apply_chat_template_error = exc
else:  # pragma: no cover - exercised indirectly
    _mlx_vlm_apply_chat_template_error = None

try:  # Optional dependency for multimodal non-stream generation.
    from mlx_vlm.generate import generate as _mlx_vlm_generate
except Exception as exc:  # pragma: no cover - optional dependency
    _mlx_vlm_generate = None
    _mlx_vlm_generate_error = exc
else:  # pragma: no cover - exercised indirectly
    _mlx_vlm_generate_error = None

try:  # Optional dependency for multimodal streaming generation.
    from mlx_vlm.generate import stream_generate as _mlx_vlm_stream_generate
except Exception as exc:  # pragma: no cover - optional dependency
    _mlx_vlm_stream_generate = None
    _mlx_vlm_stream_generate_error = exc
else:  # pragma: no cover - exercised indirectly
    _mlx_vlm_stream_generate_error = None

# ChatML special tokens to filter from non-Harmony model outputs
_CHATML_SPECIAL_TOKENS_RE = re.compile(r"<\|im_end\|>|<\|im_start\|>")


class ResponsesAdapter:
    """
    Adapter for Responses API that routes to appropriate backend.

    Supports:
    - Local MLX models via ChatGenerator
    - External providers via multi-provider routing
    - Vision models via mlx-vlm
    - Hosted tool execution (web_search, code_interpreter)
    """

    def __init__(self, model_id: str | None = None):
        """
        Initialize adapter.

        Args:
            model_id: Default model to use (can be overridden per request)
        """
        self.default_model_id = model_id

    def _get_chat_generator(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> ChatGenerator:
        """Get or create ChatGenerator for model (uses shared cache)."""
        return ChatGenerator.get_or_create(
            model_id,
            adapter_path,
            draft_model_id,
        )

    def _get_openai_adapter(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> OpenAIAdapter:
        """Get OpenAI adapter wrapping ChatGenerator."""
        generator = self._get_chat_generator(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        return OpenAIAdapter(generator)

    def _prepare_tools_for_request(
        self,
        request_tools: Any,
    ) -> tuple[list[dict[str, Any]], list[Tool] | None]:
        """Normalize tool definitions for both direct and batched text flows."""
        tool_definitions = get_tool_definitions(request_tools)
        if not tool_definitions:
            return [], None

        tools_for_request = [
            Tool(
                type="function",
                function={
                    "name": tool_def.get("name", ""),
                    "description": tool_def.get("description", ""),
                    "parameters": tool_def.get("parameters", {}),
                },
            )
            for tool_def in tool_definitions
            if tool_def.get("type") == "function"
        ]
        return tool_definitions, tools_for_request or None

    def _chat_request_runtime_params(
        self,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> dict[str, str]:
        """Build runtime override params for the chat-completions adapter surface."""
        params: dict[str, str] = {}
        if adapter_path is not None:
            params["adapter_path"] = adapter_path
        if draft_model_id is not None:
            params["draft_model"] = draft_model_id
        return params

    def _should_use_batch(self) -> bool:
        """Check if batch inference is enabled."""
        settings = get_settings()
        return settings.enable_batch_inference

    def _should_use_vlm_batch(
        self,
        normalised_body: dict[str, Any],
        *,
        stream: bool,
    ) -> bool:
        """Return True when this vision request is eligible for the VLM batch lane."""
        settings = get_settings()
        enabled = (
            settings.vlm_stream_batch_enabled if stream else settings.vlm_batch_enabled
        )
        if not enabled:
            return False

        if self._has_video_content(normalised_body):
            return False

        images = self._extract_image_inputs(normalised_body)
        return len(images) == 1

    def _multimodal_validation_error(
        self,
        normalised_body: dict[str, Any],
    ) -> str | None:
        """Return a contract violation message for unsupported media combinations."""
        if normalised_body.get("tools"):
            return "Multimodal requests with tools are not supported yet."

        media_turns = 0
        for turn in normalised_body.get("input", []):
            if not isinstance(turn, dict):
                continue
            content = turn.get("content", [])
            if isinstance(content, dict):
                content = [content]
            if not isinstance(content, list):
                continue
            if any(
                isinstance(part, dict)
                and part.get("type") in {"input_image", "input_video"}
                for part in content
            ):
                media_turns += 1

        if media_turns > 1:
            return (
                "Multimodal requests with media across multiple turns are not "
                "supported yet."
            )
        return None

    def _batch_fallback_reason(
        self,
        normalised_body: dict[str, Any],
        *,
        draft_model_id: str | None = None,
    ) -> str | None:
        """Return the reason a text request must stay off the batch lane."""
        reason: str | None = None
        if draft_model_id is not None:
            reason = "draft model"
        elif normalised_body.get("previous_response_id"):
            reason = "previous_response context"
        elif normalised_body.get("tools"):
            reason = "tools"
        elif normalised_body.get("stop"):
            reason = "custom stop"
        else:
            top_p = normalised_body.get("top_p")
            text_format = (normalised_body.get("text") or {}).get("format")
            if top_p not in (None, 1, 1.0):
                reason = "custom top_p"
            elif text_format and text_format.get("type") != "text":
                reason = "structured output"

        return reason

    def _chat_request_extra_body(
        self,
        normalised_body: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build per-request extra_body overrides for text lanes."""
        extra_body: dict[str, Any] = {}
        if normalised_body.get("previous_response_id"):
            # Reconstructed follow-ups after a multimodal turn can drift out of sync
            # with mlx_lm prompt-cache trimming on the shared VLM language tower.
            # Keep the conversation lane correct first; cached follow-ups can come
            # back once that contract is explicitly stabilized.
            extra_body["enable_prompt_cache"] = False
        return extra_body or None

    def _get_vlm_backend(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> tuple[Any, Any]:
        """Load or reuse a vision-language model under the shared endpoint surface."""
        return wrapper_cache.get_vlm_backend(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
            surface="llm",
        )

    def _require_vlm_chat_template(self):
        if _mlx_vlm_apply_chat_template is None:
            raise RuntimeError(
                "mlx-vlm is required for vision responses: "
                f"{_mlx_vlm_apply_chat_template_error}"
            )
        return _mlx_vlm_apply_chat_template

    def _require_vlm_generate(self):
        if _mlx_vlm_generate is None:
            raise RuntimeError(
                "mlx-vlm is required for vision responses: "
                f"{_mlx_vlm_generate_error}"
            )
        return _mlx_vlm_generate

    def _require_vlm_stream_generate(self):
        if _mlx_vlm_stream_generate is None:
            raise RuntimeError(
                "mlx-vlm is required for vision responses: "
                f"{_mlx_vlm_stream_generate_error}"
            )
        return _mlx_vlm_stream_generate

    def _decode_base64_image(self, data: str) -> Image.Image:
        """Decode a base64 or data URL image into a PIL image."""
        if data.startswith("data:"):
            if "," not in data:
                raise ValueError("Invalid data URL for image")
            _, data = data.split(",", 1)

        data = data.strip()
        if not data:
            raise ValueError("Empty base64 image data")

        missing = len(data) % 4
        if missing:
            data += "=" * (4 - missing)

        image_bytes = base64.b64decode(data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _content_has_image(self, content: Any) -> bool:
        if isinstance(content, dict):
            return content.get("type") == "input_image"
        if isinstance(content, list):
            return any(
                isinstance(part, dict) and part.get("type") == "input_image"
                for part in content
            )
        return False

    def _content_has_video(self, content: Any) -> bool:
        if isinstance(content, dict):
            return content.get("type") == "input_video"
        if isinstance(content, list):
            return any(
                isinstance(part, dict) and part.get("type") == "input_video"
                for part in content
            )
        return False

    def _has_image_content(self, normalised_body: dict[str, Any]) -> bool:
        for turn in normalised_body.get("input", []):
            if not isinstance(turn, dict):
                continue
            if self._content_has_image(turn.get("content")):
                return True
        return False

    def _has_video_content(self, normalised_body: dict[str, Any]) -> bool:
        for turn in normalised_body.get("input", []):
            if not isinstance(turn, dict):
                continue
            if self._content_has_video(turn.get("content")):
                return True
        return False

    def _has_vision_content(self, normalised_body: dict[str, Any]) -> bool:
        return self._has_image_content(normalised_body) or self._has_video_content(
            normalised_body
        )

    def _extract_text_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if content.get("type") in (
                "input_text",
                "output_text",
                "text",
                "reasoning_text",
            ):
                text_value = content.get("text")
                return str(text_value) if text_value is not None else ""
            text_value = content.get("text")
            return str(text_value) if text_value is not None else ""
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    if part:
                        parts.append(part)
                    continue
                if isinstance(part, dict):
                    if (
                        part.get("type")
                        in (
                            "input_text",
                            "output_text",
                            "text",
                            "reasoning_text",
                        )
                        or "text" in part
                    ):
                        text_value = part.get("text")
                        if text_value is not None:
                            parts.append(str(text_value))
                    continue
                if part is not None:
                    parts.append(str(part))
            return "\n".join(parts)
        return str(content)

    def _build_vlm_messages(
        self, normalised_body: dict[str, Any]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        preamble = collect_system_preamble(normalised_body)
        if preamble:
            messages.append({"role": "system", "content": "\n\n".join(preamble)})

        for turn in normalised_body.get("input", []):
            if not isinstance(turn, dict):
                continue

            role = str(turn.get("role", "user")).lower()
            if role not in {"system", "user", "assistant", "tool", "developer"}:
                role = "user"

            content = turn.get("content")
            text = self._extract_text_content(content)

            if text or (role == "user" and self._content_has_image(content)):
                messages.append({"role": role, "content": text})

        if not messages:
            messages.append({"role": "user", "content": ""})

        return messages

    def _extract_image_inputs(self, normalised_body: dict[str, Any]) -> list[Any]:
        images: list[Any] = []
        for turn in normalised_body.get("input", []):
            if not isinstance(turn, dict):
                continue

            content = turn.get("content", [])
            if isinstance(content, dict):
                content = [content]
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "input_image":
                    continue

                if part.get("image_base64"):
                    images.append(self._decode_base64_image(str(part["image_base64"])))
                    continue

                source = part.get("image_url")
                if isinstance(source, str) and source.startswith("data:"):
                    images.append(self._decode_base64_image(source))
                elif source:
                    images.append(source)

        return images

    def _extract_video_inputs(self, normalised_body: dict[str, Any]) -> list[str]:
        """Extract video source paths/URLs from normalised request body."""
        videos: list[str] = []
        for turn in normalised_body.get("input", []):
            if not isinstance(turn, dict):
                continue

            content = turn.get("content", [])
            if isinstance(content, dict):
                content = [content]
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "input_video":
                    continue

                source = part.get("video_url")
                if source:
                    videos.append(str(source))

        return videos

    _VLM_DEFAULT_MAX_TOKENS = 4096

    def _vlm_generation_kwargs(self, normalised_body: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        max_tokens = normalised_body.get("max_output_tokens") or normalised_body.get(
            "max_tokens"
        )
        # mlx-vlm defaults to 256 tokens which is too low for most use cases
        kwargs["max_tokens"] = (
            max_tokens if max_tokens is not None else self._VLM_DEFAULT_MAX_TOKENS
        )

        temperature = normalised_body.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = temperature

        top_p = normalised_body.get("top_p")
        if top_p is not None:
            kwargs["top_p"] = top_p

        return kwargs

    def _build_vlm_prompt(
        self,
        apply_chat_template,
        model,
        processor,
        normalised_body: dict[str, Any],
        images: list[Any],
    ) -> str:
        messages = self._build_vlm_messages(normalised_body)
        return apply_chat_template(
            processor,
            model.config,
            messages,
            add_generation_prompt=True,
            num_images=len(images),
        )

    def _prepare_vlm_stream_request(
        self,
        model_id: str,
        normalised_body: dict[str, Any],
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> (
        tuple[Any, Any, str, list[Any], dict[str, Any]]
        | tuple[None, None, None, None, dict[str, Any]]
    ):
        """Build everything needed for multimodal single-flight streaming.

        Returns either:
        - ``(model, processor, prompt, images, kwargs)`` on success, or
        - ``(None, None, None, None, error_payload)`` on contract/import/input failure.
        """
        images = self._extract_image_inputs(normalised_body)
        videos = self._extract_video_inputs(normalised_body)

        validation_error = self._multimodal_validation_error(normalised_body)
        if validation_error:
            return (
                None,
                None,
                None,
                None,
                {
                    "message": validation_error,
                    "code": "invalid_request_error",
                },
            )

        if not images and not videos:
            return (
                None,
                None,
                None,
                None,
                {
                    "message": "Vision request missing images or videos",
                    "code": "invalid_request_error",
                },
            )

        try:
            apply_chat_template = self._require_vlm_chat_template()
        except RuntimeError as exc:
            return (
                None,
                None,
                None,
                None,
                {
                    "message": f"mlx-vlm is required for vision responses: {exc}",
                    "code": "internal_error",
                },
            )

        model, processor = self._get_vlm_backend(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        kwargs = self._vlm_generation_kwargs(normalised_body)

        if videos:
            text_prompt = self._extract_text_content(
                normalised_body.get("input", [{}])[-1].get("content")
            )
            video_data = build_video_prompt_and_inputs(
                videos, text_prompt or "Describe this video.", processor, model.config
            )
            prompt = video_data.pop("prompt")
            kwargs.update(video_data)
            return model, processor, prompt, images, kwargs

        prompt = self._build_vlm_prompt(
            apply_chat_template,
            model,
            processor,
            normalised_body,
            images,
        )
        return model, processor, prompt, images, kwargs

    def _vision_start_events(
        self, make_event, response_obj: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            make_event("response.created", {"response": response_obj.copy()}),
            make_event("response.in_progress", {"response": response_obj.copy()}),
        ]

    def _vision_message_start_events(
        self, make_event, message_item_id: str, *, output_index: int = 0
    ) -> list[dict[str, Any]]:
        return [
            make_event(
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
            ),
            make_event(
                "response.content_part.added",
                {
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                },
            ),
        ]

    def _reasoning_start_events(
        self,
        make_event,
        reasoning_item_id: str,
        *,
        output_index: int = 0,
    ) -> list[dict[str, Any]]:
        return [
            make_event(
                "response.output_item.added",
                {
                    "output_index": output_index,
                    "item": {
                        "id": reasoning_item_id,
                        "type": "reasoning",
                        "summary": [],
                    },
                },
            )
        ]

    def _reasoning_done_events(
        self,
        make_event,
        reasoning_item_id: str,
        reasoning_text: str,
        *,
        output_index: int = 0,
    ) -> list[dict[str, Any]]:
        return [
            make_event(
                "response.reasoning_summary_text.done",
                {
                    "item_id": reasoning_item_id,
                    "output_index": output_index,
                    "text": reasoning_text,
                },
            ),
            make_event(
                "response.output_item.done",
                {
                    "output_index": output_index,
                    "item": {
                        "id": reasoning_item_id,
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": reasoning_text}],
                    },
                },
            ),
        ]

    def _vision_finalize_events(
        self,
        make_event,
        response_id: str,
        response_model: str,
        message_item_id: str,
        final_text: str,
        *,
        output_index: int = 0,
        reasoning_item: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        output_items: list[dict[str, Any]] = []
        if reasoning_item is not None:
            output_items.append(reasoning_item)
        output_items.append(
            {
                "id": message_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": final_text}],
            }
        )
        return [
            make_event(
                "response.output_text.done",
                {
                    "output_index": output_index,
                    "content_index": 0,
                    "text": final_text,
                },
            ),
            make_event(
                "response.content_part.done",
                {
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": final_text},
                },
            ),
            make_event(
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
            ),
            make_event(
                "response.completed",
                {
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "model": response_model,
                        "output": output_items,
                    }
                },
            ),
        ]

    async def _stream_batch_tokens(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        adapter_path: str | None = None,
    ) -> AsyncGenerator[BatchStreamChunk, None]:
        """Stream tokens through batch coordinator.

        Args:
            model_id: Model to use
            messages: Chat messages
            max_tokens: Max tokens to generate
            tools: Optional tools for function calling
            temperature: Sampling temperature
            adapter_path: Optional adapter path for exact runtime batching

        Yields:
            BatchStreamChunk for each token
        """
        settings = get_settings()

        # Get or create batch coordinator for this model
        coordinator = get_batch_coordinator(
            model_id=model_id,
            adapter_path=adapter_path,
            completion_batch_size=settings.batch_completion_size,
            prefill_batch_size=settings.batch_prefill_size,
            prefill_step_size=settings.batch_prefill_step_size,
            batch_window_ms=settings.batch_window_ms,
            max_batch_size=settings.max_batch_size,
        )

        # Build sampler config from temperature
        sampler_config = None
        if temperature is not None:
            sampler_config = {"temp": temperature}

        # Stream through coordinator
        async for chunk in coordinator.stream_request(
            messages=messages,
            max_tokens=max_tokens or 4096,
            tools=tools,
            sampler_config=sampler_config,
        ):
            yield chunk

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
        adapter_path = request.adapter_path
        draft_model_id = request.get_draft_model_id()

        if not model_id:
            return build_error_response(
                "Model not specified",
                error_code="invalid_request_error",
            )

        try:
            # Normalize request
            body = request.model_dump(exclude_none=True)
            normalised = normalise_responses_payload(body)

            # Check for media content (images or video)
            if self._has_vision_content(normalised):
                validation_error = self._multimodal_validation_error(normalised)
                if validation_error:
                    return build_error_response(
                        validation_error,
                        error_code="invalid_request_error",
                        model=request_model,
                    )
                async with endpoint_runtime_session(
                    model_id,
                    adapter_path=adapter_path,
                    draft_model_id=draft_model_id,
                ):
                    return await self._generate_vision(
                        model_id,
                        normalised,
                        request_model,
                        adapter_path=adapter_path,
                        draft_model_id=draft_model_id,
                    )

            if has_media_content(normalised):
                logger.warning(
                    "Media content detected without images; falling back to text-only"
                )

            # Text-only path
            async with endpoint_runtime_session(
                model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            ):
                return await self._generate_text(
                    model_id,
                    normalised,
                    request_model,
                    adapter_path=adapter_path,
                    draft_model_id=draft_model_id,
                )

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
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> ResponseResponse:
        """Generate text-only response using MLX, with tool support."""
        # Use request_model for response, fallback to model_id
        response_model = request_model or model_id
        # Convert to chat messages
        messages = responses_to_chat_messages(normalised_body)

        # Expand hosted tools to function definitions
        _, tools_for_request = self._prepare_tools_for_request(
            normalised_body.get("tools")
        )

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
            extra_body=self._chat_request_extra_body(normalised_body),
            **self._chat_request_runtime_params(
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            ),
        )

        # Get adapter and generate
        adapter = self._get_openai_adapter(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
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

        if (
            not (is_harmony_model(resolved_model) or is_harmony_model(model_id))
            and content_text
        ):
            parsed = parse_reasoning_like_output(content_text)
            if parsed["reasoning"]:
                content_text = parsed["final_text"]
                reasoning_text = parsed["reasoning"] or reasoning_text

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
            _provider="mlx-batch-server",
        )

    async def _generate_vision(
        self,
        model_id: str,
        normalised_body: dict[str, Any],
        request_model: str | None = None,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> ResponseResponse:
        """
        Generate response for vision/multimodal content via mlx-vlm.

        Supports both image and video inputs.  Video uses a separate pipeline
        that processes frames through the processor and passes ``video_grid_thw``
        to the generate function.
        """
        response_model = request_model or model_id

        images = self._extract_image_inputs(normalised_body)
        videos = self._extract_video_inputs(normalised_body)

        if not images and not videos:
            logger.warning(
                "Vision content detected but no media found; falling back to text-only"
            )
            return await self._generate_text(
                model_id,
                normalised_body,
                request_model,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            )

        if self._should_use_vlm_batch(normalised_body, stream=False):
            settings = get_settings()
            coordinator = get_vlm_batch_coordinator(
                model_id=model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
                batch_window_ms=settings.vlm_batch_window_ms,
                max_batch_size=settings.vlm_max_batch_size,
                group_by_shape=settings.vlm_batch_group_by_shape,
            )
            result = await coordinator.submit_request(
                messages=self._build_vlm_messages(normalised_body),
                images=images,
                max_tokens=normalised_body.get("max_output_tokens")
                or normalised_body.get("max_tokens"),
                temperature=normalised_body.get("temperature"),
                top_p=normalised_body.get("top_p"),
            )

            content_text = _CHATML_SPECIAL_TOKENS_RE.sub("", result.text or "")
            parsed = parse_reasoning_like_output(content_text)
            output_items = build_text_output(
                parsed["final_text"],
                parsed["reasoning"],
            )

            return ResponseResponse(
                id=f"resp_{uuid.uuid4().hex}",
                created_at=int(time.time()),
                model=response_model,
                status=ResponseStatus.COMPLETED,
                output=output_items,
                usage=ResponseUsage(
                    input_tokens=result.prompt_tokens,
                    output_tokens=result.generation_tokens,
                    total_tokens=result.total_tokens,
                ),
                _provider="mlx-batch-server",
            )

        apply_chat_template = self._require_vlm_chat_template()
        vlm_generate = self._require_vlm_generate()

        model, processor = self._get_vlm_backend(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        )
        gen_kwargs = self._vlm_generation_kwargs(normalised_body)

        with wrapper_cache.vlm_execution(
            model_id,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        ):
            if videos:
                # --- Video path: use mlx-vlm video pipeline ---
                text_prompt = self._extract_text_content(
                    normalised_body.get("input", [{}])[-1].get("content")
                )
                video_data = build_video_prompt_and_inputs(
                    videos,
                    text_prompt or "Describe this video.",
                    processor,
                    model.config,
                )

                result = vlm_generate(
                    model,
                    processor,
                    video_data["prompt"],
                    image=images or None,
                    **{k: v for k, v in video_data.items() if k not in ("prompt",)},
                    **gen_kwargs,
                )
            else:
                # --- Image-only path (existing) ---
                prompt = self._build_vlm_prompt(
                    apply_chat_template,
                    model,
                    processor,
                    normalised_body,
                    images,
                )

                result = vlm_generate(
                    model,
                    processor,
                    prompt,
                    image=images,
                    **gen_kwargs,
                )

        content_text = _CHATML_SPECIAL_TOKENS_RE.sub("", result.text or "")
        parsed = parse_reasoning_like_output(content_text)
        output_items = build_text_output(
            parsed["final_text"],
            parsed["reasoning"],
        )

        return ResponseResponse(
            id=f"resp_{uuid.uuid4().hex}",
            created_at=int(time.time()),
            model=response_model,
            status=ResponseStatus.COMPLETED,
            output=output_items,
            usage=ResponseUsage(
                input_tokens=result.prompt_tokens,
                output_tokens=result.generation_tokens,
                total_tokens=result.total_tokens,
            ),
            _provider="mlx-batch-server",
        )

    async def _generate_vision_stream(  # noqa: PLR0912, PLR0915
        self,
        model_id: str,
        normalised_body: dict[str, Any],
        request_model: str | None = None,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        response_model = request_model or model_id
        response_id = f"resp_{uuid.uuid4().hex}"
        message_item_id = f"msg_{uuid.uuid4().hex}"
        reasoning_item_id = f"rs_{uuid.uuid4().hex}"

        seq_num = 0

        def make_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
            nonlocal seq_num
            event = {"type": event_type, "sequence_number": seq_num, **data}
            seq_num += 1
            return event

        response_obj = {
            "id": response_id,
            "object": "response",
            "status": "in_progress",
            "model": response_model,
            "output": [],
        }
        for event in self._vision_start_events(make_event, response_obj):
            yield event

        reasoning_parser = ReasoningStreamingParser(assume_initial_reasoning=True)
        reasoning_text_parts: list[str] = []
        output_text_parts: list[str] = []
        reasoning_item_emitted = False
        reasoning_done_emitted = False
        message_item_emitted = False

        if self._should_use_vlm_batch(normalised_body, stream=True):
            settings = get_settings()
            stream_results = get_vlm_stream_coordinator(
                model_id=model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
                batch_window_ms=settings.vlm_batch_window_ms,
                max_batch_size=settings.vlm_max_batch_size,
            ).stream_request(
                messages=self._build_vlm_messages(normalised_body),
                images=self._extract_image_inputs(normalised_body),
                max_tokens=normalised_body.get("max_output_tokens")
                or normalised_body.get("max_tokens"),
                temperature=normalised_body.get("temperature"),
                top_p=normalised_body.get("top_p"),
            )
        else:
            try:
                vlm_stream_generate = self._require_vlm_stream_generate()
            except RuntimeError as exc:
                yield make_event(
                    "error",
                    {
                        "error": {
                            "message": (
                                "mlx-vlm is required for vision responses: " f"{exc}"
                            ),
                            "code": "internal_error",
                        }
                    },
                )
                return

            model, processor, prompt, images, kwargs = self._prepare_vlm_stream_request(
                model_id,
                normalised_body,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            )
            if model is None:
                yield make_event("error", {"error": kwargs})
                return

            async def _direct_stream_results():
                with wrapper_cache.vlm_execution(
                    model_id,
                    adapter_path=adapter_path,
                    draft_model_id=draft_model_id,
                ):
                    for result in vlm_stream_generate(
                        model,
                        processor,
                        prompt,
                        image=images or None,
                        **kwargs,
                    ):
                        yield result
                        await asyncio.sleep(0)

            stream_results = _direct_stream_results()

        async for result in stream_results:
            delta = _CHATML_SPECIAL_TOKENS_RE.sub("", result.text or "")
            if not delta and getattr(result, "finish_reason", None) is None:
                continue

            for event_type, clean_text in reasoning_parser.process_delta(delta):
                if not clean_text:
                    continue

                if event_type == "reasoning":
                    if not reasoning_item_emitted:
                        for event in self._reasoning_start_events(
                            make_event,
                            reasoning_item_id,
                            output_index=0,
                        ):
                            yield event
                        reasoning_item_emitted = True

                    reasoning_text_parts.append(clean_text)
                    yield make_event(
                        "response.reasoning_summary_text.delta",
                        {
                            "item_id": reasoning_item_id,
                            "output_index": 0,
                            "delta": clean_text,
                        },
                    )
                    continue

                if (
                    reasoning_item_emitted
                    and not reasoning_done_emitted
                    and reasoning_text_parts
                ):
                    for event in self._reasoning_done_events(
                        make_event,
                        reasoning_item_id,
                        "".join(reasoning_text_parts),
                        output_index=0,
                    ):
                        yield event
                    reasoning_done_emitted = True

                if not message_item_emitted:
                    for event in self._vision_message_start_events(
                        make_event,
                        message_item_id,
                        output_index=1 if reasoning_text_parts else 0,
                    ):
                        yield event
                    message_item_emitted = True

                output_text_parts.append(clean_text)
                yield make_event(
                    "response.output_text.delta",
                    {
                        "output_index": 1 if reasoning_text_parts else 0,
                        "content_index": 0,
                        "delta": clean_text,
                    },
                )

        parsed = parse_reasoning_like_output(
            reasoning_parser.full_text,
            assume_initial_reasoning=True,
        )
        final_text = parsed["final_text"]
        reasoning_text = parsed["reasoning"]

        if (
            reasoning_item_emitted
            and not reasoning_done_emitted
            and reasoning_text_parts
        ):
            for event in self._reasoning_done_events(
                make_event,
                reasoning_item_id,
                reasoning_text or "".join(reasoning_text_parts),
                output_index=0,
            ):
                yield event
            reasoning_done_emitted = True

        if not message_item_emitted:
            for event in self._vision_message_start_events(
                make_event,
                message_item_id,
                output_index=1 if reasoning_text_parts else 0,
            ):
                yield event

        reasoning_item = None
        if reasoning_text:
            reasoning_item = {
                "id": reasoning_item_id,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
            }

        for event in self._vision_finalize_events(
            make_event,
            response_id,
            response_model,
            message_item_id,
            final_text,
            output_index=1 if reasoning_text else 0,
            reasoning_item=reasoning_item,
        ):
            yield event

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
        adapter_path = request.adapter_path
        draft_model_id = request.get_draft_model_id()

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

            if self._has_vision_content(normalised):
                validation_error = self._multimodal_validation_error(normalised)
                if validation_error:
                    yield make_event(
                        "error",
                        {
                            "error": {
                                "message": validation_error,
                                "code": "invalid_request_error",
                            }
                        },
                    )
                    return
                async with endpoint_runtime_session(
                    model_id,
                    adapter_path=adapter_path,
                    draft_model_id=draft_model_id,
                ):
                    async for event in self._generate_vision_stream(
                        model_id,
                        normalised,
                        request_model,
                        adapter_path=adapter_path,
                        draft_model_id=draft_model_id,
                    ):
                        yield event
                return

            if has_media_content(normalised):
                logger.warning(
                    "Media content detected without images; streaming text-only"
                )

            async with endpoint_runtime_session(
                model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            ):
                # Convert to chat messages
                messages = responses_to_chat_messages(normalised)

                # Expand hosted tools
                tool_definitions, tools_for_request = self._prepare_tools_for_request(
                    normalised.get("tools")
                )

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
                    tool_choice=normalised.get("tool_choice"),
                    extra_body=self._chat_request_extra_body(normalised),
                    **self._chat_request_runtime_params(
                        adapter_path=adapter_path,
                        draft_model_id=draft_model_id,
                    ),
                )

                # Generate IDs (full 32-char hex for proper uniqueness)
                response_id = f"resp_{uuid.uuid4().hex}"
                reasoning_item_id = f"rs_{uuid.uuid4().hex}"
                message_item_id = f"msg_{uuid.uuid4().hex}"

                # Resolve model alias to check Harmony format (chat -> gpt-oss-120b)
                settings = get_settings()
                resolved_model = settings.get_model_alias(model_id)
                is_harmony = is_harmony_model(resolved_model) or is_harmony_model(
                    model_id
                )

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
                reasoning_parser = None if is_harmony else ReasoningStreamingParser()
                reasoning_text_parts: list[str] = []
                output_text_parts: list[str] = []
                reasoning_item_emitted = is_harmony
                reasoning_done_emitted = False
                message_item_emitted = False

                # Decide streaming mode: batch vs single
                batch_fallback_reason = self._batch_fallback_reason(
                    normalised,
                    draft_model_id=draft_model_id,
                )
                use_batch = self._should_use_batch() and batch_fallback_reason is None
                if self._should_use_batch() and batch_fallback_reason is not None:
                    logger.info(
                        "Falling back to single text lane for %s: %s",
                        model_id,
                        batch_fallback_reason,
                    )

                # Mutable container shared between token_stream closure
                # and post-stream logic for tool call capture.
                stream_result: dict[str, Any] = {
                    "tool_calls": None,
                    "finish_reason": "stop",
                }
                # Accumulate tool call deltas across chunks (name in first,
                # arguments fragments in subsequent chunks).
                _tc_accum: dict[int, dict[str, Any]] = {}

                # Create unified token stream
                async def token_stream() -> AsyncGenerator[str, None]:  # noqa: PLR0912
                    """Unified token stream supporting both batch and single modes."""
                    if use_batch:
                        # Batch mode: use coordinator
                        async for batch_chunk in self._stream_batch_tokens(
                            model_id=model_id,
                            messages=messages,
                            max_tokens=normalised.get("max_output_tokens")
                            or normalised.get("max_tokens"),
                            tools=tool_definitions or None,
                            temperature=normalised.get("temperature"),
                            adapter_path=adapter_path,
                        ):
                            if batch_chunk.text:
                                yield batch_chunk.text
                    else:
                        # Single mode: use OpenAI adapter
                        adapter = self._get_openai_adapter(
                            model_id,
                            adapter_path=adapter_path,
                            draft_model_id=draft_model_id,
                        )
                        for chunk in adapter.generate_stream(chat_request):
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]
                            delta = choice.delta
                            # Accumulate tool call deltas across chunks
                            tc_list = getattr(delta, "tool_calls", None)
                            if tc_list:
                                for tc in tc_list:
                                    idx = getattr(tc, "index", 0)
                                    fn = getattr(tc, "function", None)
                                    if idx not in _tc_accum:
                                        _tc_accum[idx] = {
                                            "index": idx,
                                            "id": getattr(tc, "id", None),
                                            "function": {"name": None, "arguments": ""},
                                        }
                                    entry = _tc_accum[idx]
                                    if getattr(tc, "id", None):
                                        entry["id"] = tc.id
                                    if fn:
                                        name = getattr(fn, "name", None)
                                        args = getattr(fn, "arguments", None)
                                        if name:
                                            entry["function"]["name"] = name
                                        if args:
                                            entry["function"]["arguments"] += args
                            fr = getattr(choice, "finish_reason", None)
                            if fr:
                                stream_result["finish_reason"] = fr
                            # Yield text content as before
                            if delta and getattr(delta, "content", None):
                                yield delta.content

                # Stream content deltas
                async for raw_content in token_stream():
                    if is_harmony and harmony_parser:
                        # Use stateful parser for channel separation
                        event_type, clean_text = harmony_parser.process_delta(
                            raw_content
                        )

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
                                    reasoning_done_emitted = True

                                # Emit message output_item.added
                                yield make_event(
                                    "response.output_item.added",
                                    {
                                        "output_index": 1
                                        if reasoning_text_parts
                                        else 0,
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
                                        "output_index": 1
                                        if reasoning_text_parts
                                        else 0,
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
                        # Non-Harmony model: translate think tags to the
                        # reasoning lane used by the Responses API.
                        clean_content = _CHATML_SPECIAL_TOKENS_RE.sub("", raw_content)
                        if not clean_content:
                            continue  # Skip empty chunks after filtering

                        if reasoning_parser is None:
                            continue

                        for event_type, parsed_text in reasoning_parser.process_delta(
                            clean_content
                        ):
                            if not parsed_text:
                                continue

                            if event_type == "reasoning":
                                if not reasoning_item_emitted:
                                    for event in self._reasoning_start_events(
                                        make_event,
                                        reasoning_item_id,
                                        output_index=0,
                                    ):
                                        yield event
                                    reasoning_item_emitted = True

                                reasoning_text_parts.append(parsed_text)
                                yield make_event(
                                    "response.reasoning_summary_text.delta",
                                    {
                                        "item_id": reasoning_item_id,
                                        "output_index": 0,
                                        "delta": parsed_text,
                                    },
                                )
                                continue

                            if (
                                reasoning_item_emitted
                                and not reasoning_done_emitted
                                and reasoning_text_parts
                            ):
                                for event in self._reasoning_done_events(
                                    make_event,
                                    reasoning_item_id,
                                    "".join(reasoning_text_parts),
                                    output_index=0,
                                ):
                                    yield event
                                reasoning_done_emitted = True

                            if not message_item_emitted:
                                yield make_event(
                                    "response.output_item.added",
                                    {
                                        "output_index": 1
                                        if reasoning_text_parts
                                        else 0,
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
                                        "output_index": 1
                                        if reasoning_text_parts
                                        else 0,
                                        "content_index": 0,
                                        "part": {"type": "output_text", "text": ""},
                                    },
                                )
                                message_item_emitted = True

                            output_text_parts.append(parsed_text)
                            yield make_event(
                                "response.output_text.delta",
                                {
                                    "output_index": 1 if reasoning_text_parts else 0,
                                    "content_index": 0,
                                    "delta": parsed_text,
                                },
                            )

                # Parse final text for Harmony models
                final_text = "".join(output_text_parts)
                reasoning_text = (
                    "".join(reasoning_text_parts) if reasoning_text_parts else None
                )

                # Finalize accumulated tool calls from stream deltas
                if _tc_accum:
                    stream_result["tool_calls"] = [
                        _tc_accum[k] for k in sorted(_tc_accum)
                    ]
                detected_tool_calls = stream_result.get("tool_calls")

                if is_harmony and harmony_parser:
                    # Final parse for clean output and tool calls
                    parsed = parse_harmony_output(harmony_parser.full_text)
                    if parsed["final_text"]:
                        final_text = parsed["final_text"]
                    if parsed["reasoning"] and not reasoning_text:
                        reasoning_text = parsed["reasoning"]
                    if parsed["tool_calls"] and not detected_tool_calls:
                        detected_tool_calls = parsed["tool_calls"]
                        tc_count = len(detected_tool_calls)
                        logger.info(f"Extracted {tc_count} tool calls from Harmony")
                elif reasoning_parser:
                    parsed = parse_reasoning_like_output(reasoning_parser.full_text)
                    if parsed["reasoning"]:
                        reasoning_text = parsed["reasoning"]
                        final_text = parsed["final_text"]

                if (
                    reasoning_item_emitted
                    and not reasoning_done_emitted
                    and reasoning_text_parts
                ):
                    for event in self._reasoning_done_events(
                        make_event,
                        reasoning_item_id,
                        reasoning_text or "".join(reasoning_text_parts),
                        output_index=0,
                    ):
                        yield event
                    reasoning_done_emitted = True

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

                # Emit function_call SSE events for detected tool calls
                fc_items: list[dict[str, Any]] = []
                if detected_tool_calls:
                    for i, tc in enumerate(detected_tool_calls):
                        fc_id = f"fc_{uuid.uuid4().hex}"
                        call_id = tc.get("id") or f"call_{uuid.uuid4().hex}"
                        func_name = tc.get("function", {}).get("name", "unknown")
                        func_args = tc.get("function", {}).get("arguments", "{}")
                        fc_output_index = output_index + 1 + i

                        fc_item = {
                            "type": "function_call",
                            "id": fc_id,
                            "call_id": call_id,
                            "name": func_name,
                            "arguments": func_args,
                            "status": "completed",
                        }
                        fc_items.append(fc_item)

                        yield make_event(
                            "response.output_item.added",
                            {
                                "output_index": fc_output_index,
                                "item": {
                                    "type": "function_call",
                                    "id": fc_id,
                                    "call_id": call_id,
                                    "name": func_name,
                                    "arguments": "",
                                },
                            },
                        )

                        yield make_event(
                            "response.function_call_arguments.delta",
                            {
                                "response_id": response_id,
                                "item_id": fc_id,
                                "output_index": fc_output_index,
                                "delta": func_args,
                            },
                        )

                        yield make_event(
                            "response.function_call_arguments.done",
                            {
                                "response_id": response_id,
                                "item_id": fc_id,
                                "call_id": call_id,
                                "output_index": fc_output_index,
                                "name": func_name,
                                "arguments": func_args,
                            },
                        )

                        yield make_event(
                            "response.output_item.done",
                            {
                                "output_index": fc_output_index,
                                "item": fc_item,
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

                # Append function_call items to output
                output_list.extend(fc_items)

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
