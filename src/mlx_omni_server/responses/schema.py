"""
OpenAI Responses API schema definitions.

Based on OpenAI's Responses API specification with extensions
for multi-provider routing.

"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResponseStatus(str, Enum):
    """Response status values."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContentPartType(str, Enum):
    """Content part types."""

    INPUT_TEXT = "input_text"
    INPUT_IMAGE = "input_image"
    INPUT_AUDIO = "input_audio"
    OUTPUT_TEXT = "output_text"
    REASONING_TEXT = "reasoning_text"


class ContentPart(BaseModel):
    """Single content part in a turn."""

    type: ContentPartType
    text: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    audio_url: str | None = None
    detail: str | None = None  # For images: "auto", "low", "high"


class InputTurn(BaseModel):
    """Input turn in conversation."""

    role: Literal["user", "assistant", "system", "developer", "tool"] = "user"
    content: str | list[ContentPart] | list[dict[str, Any]]


class OutputItem(BaseModel):
    """Output item in response."""

    id: str = Field(default_factory=lambda: f"item_{uuid.uuid4().hex[:24]}")
    type: Literal["message", "reasoning", "function_call", "function_call_output"] = (
        "message"
    )
    role: str | None = "assistant"
    status: ResponseStatus = ResponseStatus.COMPLETED
    content: list[dict[str, Any]] = Field(default_factory=list)
    # For function calls
    name: str | None = None
    call_id: str | None = None
    arguments: str | None = None
    output: str | None = None


class ResponseRequest(BaseModel):
    """
    Request body for /v1/responses endpoint.

    Compatible with OpenAI Responses API format.
    """

    model: str
    input: str | list[InputTurn] | list[dict[str, Any]]
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    output_modalities: list[str] | None = None
    instructions: str | None = None  # Alias for system_instruction
    system_instruction: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None  # Alias
    top_p: float | None = None
    stop: list[str] | str | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None  # For structured output format

    # Extensions
    provider: str | None = None  # Override auto-detection
    store: bool = True
    metadata: dict[str, Any] | None = None
    previous_response_id: str | None = None  # Chain responses

    def get_max_tokens(self) -> int | None:
        """Get max tokens from either field."""
        return self.max_output_tokens or self.max_tokens

    def get_system_instruction(self) -> str | None:
        """Get system instruction from either field."""
        return self.system_instruction or self.instructions


class ResponseUsage(BaseModel):
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: dict[str, int] | None = None
    output_tokens_details: dict[str, int] | None = None


class ResponseResponse(BaseModel):
    """
    Response body for /v1/responses endpoint.

    Compatible with OpenAI Responses API format.
    """

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    status: ResponseStatus = ResponseStatus.COMPLETED
    output: list[OutputItem] = Field(default_factory=list)
    usage: ResponseUsage = Field(default_factory=ResponseUsage)
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None

    # Extensions
    _provider: str | None = None
    _api_version: str = "responses-v1"

    model_config = ConfigDict(extra="allow")


class ResponseStreamEvent(BaseModel):
    """Single event in SSE stream."""

    type: str
    response: dict[str, Any] | None = None
    output_index: int | None = None
    item: dict[str, Any] | None = None
    content_index: int | None = None
    part: dict[str, Any] | None = None
    delta: str | None = None
    error: dict[str, Any] | None = None


# Utility functions for building responses


def build_text_output(text: str, reasoning: str | None = None) -> list[OutputItem]:
    """Build output items from text response."""
    items: list[OutputItem] = []

    if reasoning:
        items.append(
            OutputItem(
                id=f"rs_{uuid.uuid4().hex[:24]}",
                type="reasoning",
                role="assistant",
                status=ResponseStatus.COMPLETED,
                content=[{"type": "reasoning_text", "text": reasoning}],
            )
        )

    items.append(
        OutputItem(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            type="message",
            role="assistant",
            status=ResponseStatus.COMPLETED,
            content=[{"type": "output_text", "text": text}],
        )
    )

    return items


def build_error_response(
    error_message: str,
    error_code: str = "internal_error",
    model: str = "unknown",
) -> ResponseResponse:
    """Build error response."""
    return ResponseResponse(
        model=model,
        status=ResponseStatus.FAILED,
        output=[],
        error={
            "message": error_message,
            "type": "error",
            "code": error_code,
        },
    )
