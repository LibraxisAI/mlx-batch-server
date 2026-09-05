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
    """What a client may *ask* for."""

    AUTO = "auto"
    STANDARD_ONLY = "standard_only"


class ResponseServiceTier(StrEnum):
    """What was actually *served*.

    A different vocabulary from :class:`ServiceTier` on purpose: the request
    names a routing preference, the response names the capacity lane that ran
    the turn. Collapsing the two would let ``standard_only`` be echoed back as
    though it were a delivered tier.
    """

    STANDARD = "standard"
    PRIORITY = "priority"
    BATCH = "batch"


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
    # Required, exactly as the official SDK models it. There is no default: a
    # block opened on the wire says ``signature=""`` out loud (the signature
    # arrives later as a signature_delta), while a terminal block has to be
    # handed a real one. A defaulted field let an unsigned terminal thinking
    # block look well-formed, which is the hole W3-AB closes.
    signature: str


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


class CacheControl(BaseModel):
    """An Anthropic prompt-cache breakpoint.

    Represented so the capability preflight can refuse it at its exact block
    location. Naming it here is not a claim that this runtime caches.
    """

    model_config = _STRICT

    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = None


class RequestTextBlock(BaseModel):
    model_config = _STRICT

    type: Literal["text"] = "text"
    text: str
    cache_control: CacheControl | None = None


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
    cache_control: CacheControl | None = None


class RequestThinkingBlock(BaseModel):
    model_config = _STRICT

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""
    cache_control: CacheControl | None = None


class RequestRedactedThinkingBlock(BaseModel):
    """Opaque reasoning returned by Anthropic and replayed by a client.

    Represented so a continuation carrying redacted reasoning is refused at
    its exact location instead of being dropped on the way to inference.
    """

    model_config = _STRICT

    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str
    cache_control: CacheControl | None = None


class RequestToolUseBlock(BaseModel):
    model_config = _STRICT

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(..., min_length=1)
    name: str
    input: dict[str, Any]
    cache_control: CacheControl | None = None


class RequestServerToolUseBlock(BaseModel):
    """An Anthropic-hosted server tool invocation, represented then refused."""

    model_config = _STRICT

    type: Literal["server_tool_use"] = "server_tool_use"
    id: str = Field(..., min_length=1)
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    cache_control: CacheControl | None = None


class RequestContainerUploadBlock(BaseModel):
    """A code-execution container upload, represented then refused."""

    model_config = _STRICT

    type: Literal["container_upload"] = "container_upload"
    file_id: str = Field(..., min_length=1)
    cache_control: CacheControl | None = None


class CitationsConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = False


class DocumentSource(BaseModel):
    """Official document source union, accepted then mapped or refused.

    ``type="content"`` nested documents are parsed so the mapper can fail
    closed with a named capability error instead of a generic validation miss.
    """

    model_config = _STRICT

    type: Literal["base64", "url", "file", "text", "content"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    content: list[RequestTextBlock] | None = None


class RequestDocumentBlock(BaseModel):
    model_config = _STRICT

    type: Literal["document"] = "document"
    source: DocumentSource
    title: str | None = None
    context: str | None = None
    citations: CitationsConfig | None = None
    cache_control: CacheControl | None = None


class RequestSearchResultBlock(BaseModel):
    model_config = _STRICT

    type: Literal["search_result"] = "search_result"
    source: str
    title: str
    content: list[RequestTextBlock]
    citations: CitationsConfig | None = None
    cache_control: CacheControl | None = None


class RequestToolReferenceBlock(BaseModel):
    model_config = _STRICT

    type: Literal["tool_reference"] = "tool_reference"
    tool_name: str
    cache_control: CacheControl | None = None


ToolResultContent = Annotated[
    Union[
        RequestTextBlock,
        RequestImageBlock,
        RequestDocumentBlock,
        RequestSearchResultBlock,
        RequestToolReferenceBlock,
    ],
    Field(discriminator="type"),
]


class RequestToolResultBlock(BaseModel):
    model_config = _STRICT

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(..., min_length=1)
    content: Union[str, list[ToolResultContent]] = ""
    is_error: bool = False
    cache_control: CacheControl | None = None


class RequestWebSearchToolResultBlock(BaseModel):
    """A hosted web-search result replayed by a client, represented then refused."""

    model_config = _STRICT

    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: str = Field(..., min_length=1)
    content: Union[str, list[dict[str, Any]], dict[str, Any], None] = None
    cache_control: CacheControl | None = None


#: Every official request-content discriminator W3 must be able to *name*.
#: Membership here is representation, never a capability claim: the profile
#: in ``capabilities`` decides which of these may execute.
RequestContentBlock = Annotated[
    Union[
        RequestTextBlock,
        RequestImageBlock,
        RequestDocumentBlock,
        RequestSearchResultBlock,
        RequestThinkingBlock,
        RequestRedactedThinkingBlock,
        RequestToolUseBlock,
        RequestToolResultBlock,
        RequestServerToolUseBlock,
        RequestWebSearchToolResultBlock,
        RequestContainerUploadBlock,
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
    cache_control: CacheControl | None = None


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
    """A tool definition on the wire.

    ``type`` is deliberately open: an Anthropic-hosted server tool carries a
    discriminator such as ``web_search_20250305``, and representing it is
    what lets the capability preflight refuse it by name instead of emitting
    a generic union mismatch. ``input_schema`` is optional for the same
    reason — hosted definitions omit it — and is required back by the
    preflight for every custom tool.
    """

    model_config = _STRICT

    name: str = Field(..., max_length=200, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str | None = None
    input_schema: ToolInputSchema | None = None
    type: str | None = None
    cache_control: CacheControl | None = None


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


class OutputConfig(BaseModel):
    """Structured-output and effort controls, represented then refused.

    Both wire placements of ``effort`` are named — nested here and top-level
    on the request — so neither shape can slip past preflight unclassified.
    """

    model_config = _STRICT

    format: dict[str, Any] | None = None
    effort: str | None = None


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
    #: The capacity lane that actually served the turn. Reported here — the
    #: same place the official API reports it — so a client reads the
    #: delivered tier instead of inferring it from what it asked for.
    service_tier: ResponseServiceTier | None = None


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

    # Represented so preflight can refuse each by name. Presence in this
    # model is not a capability claim; see ``capabilities``.
    container: Union[str, dict[str, Any], None] = None
    inference_geo: str | None = None
    output_config: OutputConfig | None = None
    effort: str | None = None


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
    "CacheControl",
    "CitationsConfig",
    "ContentBlock",
    "ContentBlockDeltaBody",
    "ContentBlockDeltaEvent",
    "ContentBlockStartEvent",
    "ContentBlockStopEvent",
    "DocumentSource",
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
    "OutputConfig",
    "PingEvent",
    "RequestContainerUploadBlock",
    "RequestContentBlock",
    "RequestDocumentBlock",
    "RequestImageBlock",
    "RequestRedactedThinkingBlock",
    "RequestSearchResultBlock",
    "RequestServerToolUseBlock",
    "RequestTextBlock",
    "RequestThinkingBlock",
    "RequestToolReferenceBlock",
    "RequestToolResultBlock",
    "RequestToolUseBlock",
    "RequestWebSearchToolResultBlock",
    "ResponseServiceTier",
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
    "ToolResultContent",
    "ToolUseBlock",
    "Usage",
]
