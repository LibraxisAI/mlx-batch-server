"""Anthropic Messages API Schema Definitions

This module defines Pydantic models for the Anthropic Messages API,
following the official API specification.
"""

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# Basic Enums and Types
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


class ServiceTier(str, Enum):
    AUTO = "auto"
    STANDARD_ONLY = "standard_only"


class ToolChoiceType(str, Enum):
    AUTO = "auto"
    ANY = "any"
    NONE = "none"
    TOOL = "tool"


# Content Blocks
class TextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """Thinking content block for extended reasoning."""

    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolUseBlock(BaseModel):
    """Tool use content block."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ImageBlock(BaseModel):
    """Image content block (placeholder for future implementation)."""

    type: Literal["image"] = "image"
    source: dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result content block."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, list[Union[TextBlock, ImageBlock]]]
    is_error: bool | None = False


# Content block union type
ContentBlock = Union[
    TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock
]


# Tool Definitions
class ToolInputSchema(BaseModel):
    """JSON schema for tool input."""

    type: str = "object"
    properties: dict[str, Any] | None = None
    required: list[str] | None = None


class AnthropicTool(BaseModel):
    """Tool definition for Anthropic API."""

    name: str = Field(..., max_length=200, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str | None = None
    input_schema: ToolInputSchema


class ToolChoiceAuto(BaseModel):
    """Automatic tool choice."""

    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: bool | None = False


class ToolChoiceAny(BaseModel):
    """Use any available tool."""

    type: Literal["any"] = "any"
    disable_parallel_tool_use: bool | None = False


class ToolChoiceNone(BaseModel):
    """Don't use any tools."""

    type: Literal["none"] = "none"


class ToolChoiceTool(BaseModel):
    """Use specific tool."""

    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: bool | None = False


ToolChoice = Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceNone, ToolChoiceTool]


# Thinking Configuration
class ThinkingConfigEnabled(BaseModel):
    """Enabled thinking configuration."""

    type: Literal["enabled"] = "enabled"
    budget_tokens: int = Field(..., ge=1024)


class ThinkingConfigDisabled(BaseModel):
    """Disabled thinking configuration."""

    type: Literal["disabled"] = "disabled"


ThinkingConfig = Union[ThinkingConfigEnabled, ThinkingConfigDisabled]


# Request Messages
class RequestTextBlock(BaseModel):
    """Text block in request."""

    type: Literal["text"] = "text"
    text: str


class RequestImageBlock(BaseModel):
    """Image block in request (base64 format)."""

    type: Literal["image"] = "image"
    source: dict[str, Any]  # Simplified for now


class RequestToolUseBlock(BaseModel):
    """Tool use block in request."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class RequestToolResultBlock(BaseModel):
    """Tool result block in request."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, list[Union[RequestTextBlock, RequestImageBlock]]]
    is_error: bool | None = False


RequestContentBlock = Union[
    RequestTextBlock, RequestImageBlock, RequestToolUseBlock, RequestToolResultBlock
]


class InputMessage(BaseModel):
    """Input message for Messages API."""

    role: MessageRole
    content: Union[str, list[RequestContentBlock]]


# System Prompt
class SystemTextBlock(BaseModel):
    """System text block."""

    type: Literal["text"] = "text"
    text: str


SystemPrompt = Union[str, list[SystemTextBlock]]


# Metadata
class Metadata(BaseModel):
    """Request metadata."""

    user_id: str | None = Field(None, max_length=256)


# Usage Statistics
class Usage(BaseModel):
    """Usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


# Main Request Model
class MessagesRequest(BaseModel):
    """Anthropic Messages API request."""

    # Required fields
    model: str = Field(..., max_length=256, min_length=1)
    messages: list[InputMessage]
    max_tokens: int = Field(..., ge=1)

    # Optional fields
    system: SystemPrompt | None = None
    temperature: float | None = Field(None, ge=0, le=1)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=0)
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    tools: list[AnthropicTool] | None = None
    tool_choice: ToolChoice | None = None
    thinking: ThinkingConfig | None = None
    metadata: Metadata | None = None
    service_tier: ServiceTier | None = None

    # Allow extra fields for compatibility
    class Config:
        extra = "allow"

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @field_validator("top_p")
    def validate_top_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Top_p must be between 0 and 1")
        return v


# Main Response Model
class MessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: StopReason | None = None
    stop_sequence: str | None = None
    usage: Usage
    container: dict[str, Any] | None = None


# Streaming Models
class StreamEventType(str, Enum):
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    PING = "ping"


class StreamDelta(BaseModel):
    """Delta object for streaming."""

    type: str | None = None
    text: str | None = None
    thinking: str | None = None
    partial_json: str | None = None
    signature: str | None = None
    stop_reason: StopReason | None = None
    stop_sequence: str | None = None
    usage: Usage | None = None


class MessageStreamEvent(BaseModel):
    """Base streaming event."""

    type: StreamEventType

    # Event-specific data
    message: MessagesResponse | None = None
    delta: StreamDelta | None = None
    content_block: ContentBlock | None = None
    index: int | None = None
    usage: Usage | None = None


# Error Models (for compatibility)
class AnthropicError(BaseModel):
    """Anthropic API error."""

    type: str
    message: str


class ErrorResponse(BaseModel):
    """Error response."""

    type: Literal["error"] = "error"
    error: AnthropicError
