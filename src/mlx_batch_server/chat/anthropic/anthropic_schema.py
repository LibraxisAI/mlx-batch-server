"""Anthropic Messages API schema.

Every model here fails closed: unknown fields are rejected rather than
absorbed, so a client is never told a capability was honoured when it was
silently dropped. Streaming events are modelled one class per wire event
instead of a single optional-field carrier, which keeps the emitted JSON
identical in shape to the documented protocol without ``exclude_none`` tricks.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid")


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


class ServiceTier(StrEnum):
    AUTO = "auto"
    STANDARD_ONLY = "standard_only"


# ---------------------------------------------------------------------------
# Response content blocks
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    """Assistant text content."""

    model_config = _STRICT

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    """Extended-reasoning content, never duplicated into a text block."""

    model_config = _STRICT

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class ToolUseBlock(BaseModel):
    """A resolved tool call with fully assembled input."""

    model_config = _STRICT

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


ContentBlock = Annotated[
    Union[TextBlock, ThinkingBlock, ToolUseBlock],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Request content blocks
# ---------------------------------------------------------------------------


class RequestTextBlock(BaseModel):
    model_config = _STRICT

    type: Literal["text"] = "text"
    text: str


class ImageSource(BaseModel):
    """Image payload descriptor.

    Accepted by the schema so the request can be *named* accurately, then
    rejected by the request mapper with an explicit unsupported-capability
    error. Silent truncation of an image would misreport what was inferred.
    """

    model_config = _STRICT

    type: Literal["base64", "url", "file"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None
    file_id: str | None = None


class RequestImageBlock(BaseModel):
    model_config = _STRICT

    type: Literal["image"] = "image"
    source: ImageSource


class RequestThinkingBlock(BaseModel):
    model_config = _STRICT

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""


class RequestToolUseBlock(BaseModel):
    model_config = _STRICT

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


ToolResultContent = Union[RequestTextBlock, RequestImageBlock]


class RequestToolResultBlock(BaseModel):
    model_config = _STRICT

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, list[ToolResultContent]] = ""
    is_error: bool = False


RequestContentBlock = Annotated[
    Union[
        RequestTextBlock,
        RequestImageBlock,
        RequestThinkingBlock,
        RequestToolUseBlock,
        RequestToolResultBlock,
    ],
    Field(discriminator="type"),
]


class InputMessage(BaseModel):
    model_config = _STRICT

    role: MessageRole
    content: Union[str, list[RequestContentBlock]]


class SystemTextBlock(BaseModel):
    model_config = _STRICT

    type: Literal["text"] = "text"
    text: str


SystemPrompt = Union[str, list[SystemTextBlock]]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ToolInputSchema(BaseModel):
    """JSON schema for a client-defined tool.

    Kept open (``extra="allow"``) on purpose: this is an arbitrary JSON Schema
    document supplied by the caller, not an Anthropic-defined envelope.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["object"] = "object"
    properties: dict[str, Any] | None = None
    required: list[str] | None = None


class AnthropicTool(BaseModel):
    """A client-defined (custom) tool.

    Anthropic-hosted server tools carry a ``type`` discriminator such as
    ``web_search_20250305``; those are rejected by the request mapper because
    this runtime executes no hosted tools.
    """

    model_config = _STRICT

    name: str = Field(..., max_length=200, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str | None = None
    input_schema: ToolInputSchema
    type: Literal["custom"] | None = None


class ToolChoiceAuto(BaseModel):
    model_config = _STRICT

    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: bool = False


class ToolChoiceAny(BaseModel):
    model_config = _STRICT

    type: Literal["any"] = "any"
    disable_parallel_tool_use: bool = False


class ToolChoiceNone(BaseModel):
    model_config = _STRICT

    type: Literal["none"] = "none"


class ToolChoiceTool(BaseModel):
    model_config = _STRICT

    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: bool = False


ToolChoice = Annotated[
    Union[ToolChoiceAuto, ToolChoiceAny, ToolChoiceNone, ToolChoiceTool],
    Field(discriminator="type"),
]


class ThinkingConfigEnabled(BaseModel):
    model_config = _STRICT

    type: Literal["enabled"] = "enabled"
    budget_tokens: int = Field(..., ge=1024)


class ThinkingConfigDisabled(BaseModel):
    model_config = _STRICT

    type: Literal["disabled"] = "disabled"


ThinkingConfig = Annotated[
    Union[ThinkingConfigEnabled, ThinkingConfigDisabled],
    Field(discriminator="type"),
]


class Metadata(BaseModel):
    model_config = _STRICT

    user_id: str | None = Field(None, max_length=256)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    """Full usage block, as carried by ``message_start`` and the final message."""

    model_config = _STRICT

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class MessageDeltaUsage(BaseModel):
    """Cumulative usage carried by ``message_delta``.

    ``output_tokens`` is the running total for the whole message, not the
    delta since the previous event.
    """

    model_config = _STRICT

    output_tokens: int = 0
    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


# ---------------------------------------------------------------------------
# Request / response envelopes
# ---------------------------------------------------------------------------


class MessagesRequest(BaseModel):
    """Anthropic Messages request.

    ``extra="forbid"`` is the contract: a field this runtime does not
    understand is a client-visible error, never a silent no-op.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., max_length=256, min_length=1)
    messages: list[InputMessage]
    max_tokens: int = Field(..., ge=1)

    system: SystemPrompt | None = None
    temperature: float | None = Field(None, ge=0, le=1)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=0)
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[AnthropicTool] | None = None
    tool_choice: ToolChoice | None = None
    thinking: ThinkingConfig | None = None
    metadata: Metadata | None = None
    service_tier: ServiceTier | None = None


class MessagesResponse(BaseModel):
    """Terminal Anthropic message envelope."""

    model_config = _STRICT

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: StopReason | None = None
    stop_sequence: str | None = None
    usage: Usage


# ---------------------------------------------------------------------------
# Streaming events — one class per documented wire event
# ---------------------------------------------------------------------------


class StreamEventType(StrEnum):
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    PING = "ping"
    ERROR = "error"


class TextDeltaBody(BaseModel):
    model_config = _STRICT

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDeltaBody(BaseModel):
    model_config = _STRICT

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class SignatureDeltaBody(BaseModel):
    model_config = _STRICT

    type: Literal["signature_delta"] = "signature_delta"
    signature: str


class InputJsonDeltaBody(BaseModel):
    """Partial JSON for a streaming tool call.

    The concatenation of every ``partial_json`` for one block is exactly the
    final tool arguments string — emitted once, never replayed.
    """

    model_config = _STRICT

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


ContentBlockDeltaBody = Annotated[
    Union[TextDeltaBody, ThinkingDeltaBody, SignatureDeltaBody, InputJsonDeltaBody],
    Field(discriminator="type"),
]


class MessageStartEvent(BaseModel):
    model_config = _STRICT

    type: Literal["message_start"] = "message_start"
    message: MessagesResponse


class ContentBlockStartEvent(BaseModel):
    model_config = _STRICT

    type: Literal["content_block_start"] = "content_block_start"
    index: int = Field(..., ge=0)
    content_block: ContentBlock


class ContentBlockDeltaEvent(BaseModel):
    model_config = _STRICT

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int = Field(..., ge=0)
    delta: ContentBlockDeltaBody


class ContentBlockStopEvent(BaseModel):
    model_config = _STRICT

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int = Field(..., ge=0)


class MessageDeltaBody(BaseModel):
    model_config = _STRICT

    stop_reason: StopReason | None = None
    stop_sequence: str | None = None


class MessageDeltaEvent(BaseModel):
    model_config = _STRICT

    type: Literal["message_delta"] = "message_delta"
    delta: MessageDeltaBody
    usage: MessageDeltaUsage


class MessageStopEvent(BaseModel):
    model_config = _STRICT

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    model_config = _STRICT

    type: Literal["ping"] = "ping"


class StreamErrorBody(BaseModel):
    model_config = _STRICT

    type: str
    message: str


class StreamErrorEvent(BaseModel):
    model_config = _STRICT

    type: Literal["error"] = "error"
    error: StreamErrorBody


AnthropicStreamEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    StreamErrorEvent,
]


__all__ = [
    "AnthropicStreamEvent",
    "AnthropicTool",
    "ContentBlock",
    "ContentBlockDeltaBody",
    "ContentBlockDeltaEvent",
    "ContentBlockStartEvent",
    "ContentBlockStopEvent",
    "ImageSource",
    "InputJsonDeltaBody",
    "InputMessage",
    "MessageDeltaBody",
    "MessageDeltaEvent",
    "MessageDeltaUsage",
    "MessageRole",
    "MessageStartEvent",
    "MessageStopEvent",
    "MessagesRequest",
    "MessagesResponse",
    "Metadata",
    "PingEvent",
    "RequestContentBlock",
    "RequestImageBlock",
    "RequestTextBlock",
    "RequestThinkingBlock",
    "RequestToolResultBlock",
    "RequestToolUseBlock",
    "ServiceTier",
    "SignatureDeltaBody",
    "StopReason",
    "StreamErrorBody",
    "StreamErrorEvent",
    "StreamEventType",
    "SystemPrompt",
    "SystemTextBlock",
    "TextBlock",
    "TextDeltaBody",
    "ThinkingBlock",
    "ThinkingConfig",
    "ThinkingConfigDisabled",
    "ThinkingConfigEnabled",
    "ThinkingDeltaBody",
    "ToolChoice",
    "ToolChoiceAny",
    "ToolChoiceAuto",
    "ToolChoiceNone",
    "ToolChoiceTool",
    "ToolInputSchema",
    "ToolUseBlock",
    "Usage",
]
