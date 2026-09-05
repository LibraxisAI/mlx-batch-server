"""Project typed runtime turn events onto the Anthropic Messages protocol.

This is the only place that knows how a generation turn becomes Anthropic
wire shapes. It owns content-block indices, cumulative usage and stop-reason
mapping for *both* transports, so the streaming lifecycle and the non-stream
envelope can never disagree about what the model produced.

The projector consumes ``mlx_batch_server.runtime.events`` — the shared,
protocol-neutral event family — and never reaches into another protocol's
internals. Provider semantics are mapped, not flattened.

Two truths are *given* to the projector rather than inferred by it:

``thinking``
    Whether this turn may put reasoning on the Anthropic wire at all, and who
    signs it. A runtime reasoning event is evidence that the runtime reasoned;
    it is not evidence that the client asked for Anthropic extended thinking,
    nor that anything can sign the block. Both are capability decisions, and
    they are made once — by ``capabilities`` — and carried here.
``service_tier``
    The capacity lane that actually served the turn, reported as delivered
    rather than echoed back from what the request preferred.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from mlx_batch_server.runtime.events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartCompleted,
    ContentPartStarted,
    HostedCallCompleted,
    HostedCallProgress,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
    OutputItemCompleted,
    ProgressUpdate,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnCancelled,
    TurnCompleted,
    TurnEvent,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)
from mlx_batch_server.tools.hosted import HOSTED_ERROR_CODES

from .anthropic_schema import (
    AnthropicStreamEvent,
    CitationCharLocation,
    CitationsDeltaBody,
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    DocumentBlock,
    InputJsonDeltaBody,
    MessageDeltaBody,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessagesResponse,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    PlainTextSource,
    ResponseServiceTier,
    ServerToolUseBlock,
    SignatureDeltaBody,
    StopReason,
    StreamErrorBody,
    StreamErrorEvent,
    TextBlock,
    TextDeltaBody,
    ThinkingBlock,
    ThinkingDeltaBody,
    ToolUseBlock,
    Usage,
    WebFetchErrorCode,
    WebFetchResultBlock,
    WebFetchToolResultBlock,
    WebFetchToolResultErrorBlock,
)
from .errors import AnthropicAPIError

_TEXT = "text"
_THINKING = "thinking"
_TOOL_USE = "tool_use"
_SERVER_TOOL_USE = "server_tool_use"
_WEB_FETCH_RESULT = "web_fetch_tool_result"

#: Runtime finish reasons that mean the output was truncated. Truncation is
#: resolved before tool use: a tool call cut off mid-arguments is a
#: ``max_tokens`` turn, not a ``tool_use`` turn.
_TRUNCATION_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})
_TOOL_REASONS = frozenset({"tool_calls", "tool_use", "function_call"})
_REFUSAL_REASONS = frozenset({"refusal", "content_filter"})
_PAUSE_REASONS = frozenset({"pause_turn", "pause"})
_STOP_SEQUENCE_REASONS = frozenset({"stop_sequence", "stop_sequences"})


@dataclass(frozen=True, slots=True)
class ThinkingSignature:
    """One integrity signature issued by a named owner for one thinking block.

    Constructing this type is the *only* way a thinking block reaches the
    wire, and it refuses to exist without both halves. That is deliberate:
    "unsigned thinking" then has no representation to leak through, rather
    than being a rule someone has to remember to check.
    """

    owner: str
    value: str

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise ValueError("a thinking signature must name the owner that issued it")
        if not self.value.strip():
            raise ValueError("a thinking signature must not be empty")


@runtime_checkable
class ThinkingSignatureOwner(Protocol):
    """Whoever can vouch for the integrity of a thinking block.

    No such owner exists on this runtime today, and the local capability
    profile says so. The seam is named rather than implemented so the
    truthful tier is a *decision* with a place to change, not an omission.
    """

    def sign_thinking(
        self, *, message_id: str, index: int, thinking: str
    ) -> ThinkingSignature:
        """Issue the signature for one completed thinking block."""
        ...


@dataclass(frozen=True, slots=True)
class ThinkingProjection:
    """The admitted decision about Anthropic thinking output for one turn.

    ``refused()`` is the default everywhere. A projector that was told
    nothing emits nothing: reasoning-channel runtime events are dropped
    before any block exists, so no ``thinking``, ``thinking_delta``,
    ``signature_delta`` or ``redacted_thinking`` can appear on the wire.
    """

    signature_owner: ThinkingSignatureOwner | None = None

    @classmethod
    def refused(cls) -> ThinkingProjection:
        """No thinking on the wire — the truthful local tier."""

        return cls()

    @classmethod
    def signed_by(cls, owner: ThinkingSignatureOwner) -> ThinkingProjection:
        """Admit thinking output, vouched for by ``owner``."""

        return cls(signature_owner=owner)

    @property
    def admitted(self) -> bool:
        return self.signature_owner is not None

    def sign(self, *, message_id: str, index: int, thinking: str) -> ThinkingSignature:
        """Obtain the signature for one block, or fail before exposing it."""

        owner = self.signature_owner
        if owner is None:
            raise AnthropicAPIError(
                "a thinking block was assembled without an admitted signature "
                "owner; the turn fails rather than emitting unsigned reasoning",
                error_type="api_error",
            )
        signature = owner.sign_thinking(
            message_id=message_id, index=index, thinking=thinking
        )
        if not isinstance(signature, ThinkingSignature):
            # A signature owner that returns something else has not signed
            # anything. Refuse it here rather than str()-ing it onto the wire.
            raise AnthropicAPIError(
                "the admitted thinking signature owner returned no "
                f"{ThinkingSignature.__name__}",
                error_type="api_error",
            )
        return signature


@dataclass
class _Block:
    """One Anthropic content block under construction."""

    index: int
    kind: str
    streamed_text: str = ""
    streamed_signature: str = ""
    call_id: str = ""
    name: str = ""
    streamed_arguments: str = ""
    pending_arguments: str = ""
    started: bool = False
    closed: bool = False
    fixed_content: ContentBlock | None = None
    citations: list[CitationCharLocation] = field(default_factory=list)


@dataclass
class _HostedCall:
    """Protocol projection state for one neutral hosted call receipt."""

    call_id: str
    item_id: str
    tool_use_id: str
    action: dict[str, Any]
    result: Mapping[str, Any] | None = None
    document_index: int | None = None
    completed: bool = False
    result_emitted: bool = False


@dataclass
class ProjectionState:
    """Snapshot of everything the terminal envelope needs."""

    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason | None = None
    stop_sequence: str | None = None
    finish_reason: str | None = None


class AnthropicMessageProjector:
    """Turn one typed runtime turn into Anthropic Messages events."""

    def __init__(
        self,
        *,
        message_id: str,
        model_alias: str,
        initial_usage: Usage | None = None,
        thinking: ThinkingProjection | None = None,
        service_tier: ResponseServiceTier = ResponseServiceTier.STANDARD,
    ) -> None:
        if not message_id.strip():
            raise ValueError("message_id must not be empty")
        if not model_alias.strip():
            raise ValueError("model_alias must not be empty")
        self._message_id = message_id
        # Omitted means refused. A caller that forgets to pass a decision gets
        # the truthful tier, not the permissive one.
        self._thinking = thinking or ThinkingProjection.refused()
        # The lane that actually served this turn. This process runs exactly
        # one, and reports it rather than echoing the requested preference.
        self._service_tier = service_tier
        # The public alias the caller asked for. The runtime's resolved
        # physical model identity is deliberately never substituted here: the
        # ``model`` a client sees at message_start is the same one it sees in
        # the terminal envelope.
        self._model_alias = model_alias
        self._state = ProjectionState(usage=self._stamp_tier(initial_usage or Usage()))
        self._blocks: dict[tuple[Any, ...], _Block] = {}
        self._order: list[_Block] = []
        self._hosted_calls: dict[str, _HostedCall] = {}
        self._document_count = 0
        self._next_index = 0
        self._started = False
        self._stopped = False
        self._failed: StreamErrorBody | None = None

    # -- public surface ---------------------------------------------------

    @property
    def message_id(self) -> str:
        return self._message_id

    @property
    def model_alias(self) -> str:
        return self._model_alias

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def failure(self) -> StreamErrorBody | None:
        return self._failed

    @property
    def usage(self) -> Usage:
        return self._state.usage

    @property
    def emits_thinking(self) -> bool:
        """Whether this turn may put thinking on the Anthropic wire at all."""

        return self._thinking.admitted

    @property
    def service_tier(self) -> ResponseServiceTier:
        """The capacity lane this turn was actually served by."""

        return self._service_tier

    def observe(  # noqa: PLR0911, PLR0912 - explicit typed event dispatcher
        self, event: TurnEvent
    ) -> tuple[AnthropicStreamEvent, ...]:
        """Project one runtime event onto zero or more Anthropic events."""

        if self._stopped:
            # Cancellation, outer deadline, failure and normal completion are
            # all hard projection barriers. A misbehaving producer cannot
            # append a hidden continuation or a late hosted result afterward.
            return ()
        if isinstance(event, TurnStarted):
            return self._on_turn_started()
        if isinstance(event, ContentPartStarted):
            return self._on_content_part_started(event)
        if isinstance(event, TextDelta):
            return self._on_text_delta(event)
        if isinstance(event, ReasoningDelta):
            return self._on_reasoning_delta(event)
        if isinstance(event, TextCompleted | ContentPartCompleted | ReasoningCompleted):
            return self._on_content_completed(event)
        if isinstance(event, ToolDelta):
            return self._on_tool_delta(event)
        if isinstance(event, ToolCompleted):
            return self._on_tool_completed(
                key=self._tool_key(event.item_id, event.index),
                call_id=event.call_id,
                name=event.name,
                arguments=event.arguments,
            )
        if isinstance(event, HostedCallStarted):
            return self._on_hosted_call_started(event)
        if isinstance(event, HostedCallProgress):
            return ()
        if isinstance(event, HostedCallResult):
            return self._on_hosted_call_result(event)
        if isinstance(event, HostedCallCompleted):
            return self._on_hosted_call_completed(event)
        if isinstance(event, HostedCitation):
            return self._on_hosted_citation(event)
        if isinstance(event, OutputItemCompleted):
            return self._on_output_item_completed(event)
        if isinstance(event, UsageUpdate):
            self._absorb_usage(event)
            return ()
        if isinstance(event, ProgressUpdate):
            # A progress phase carries no protocol content. Emitting the
            # documented ``ping`` keeps the connection warm without inventing
            # a shape or leaking internal phase names to the client.
            return (PingEvent(),)
        if isinstance(event, TurnCompleted):
            return self._on_turn_completed(event)
        if isinstance(event, TurnFailed):
            return self._on_turn_failed(event.code, event.error)
        if isinstance(event, TurnCancelled):
            return self._on_turn_failed("api_error", f"turn cancelled: {event.reason}")
        return ()

    def observe_started(self) -> None:
        """Record that message_start was already emitted by the transport."""

        self._started = True

    def fail(self, code: str, message: str) -> tuple[AnthropicStreamEvent, ...]:
        """Project a failure onto the documented Anthropic ``error`` event."""

        return self._on_turn_failed(code, message)

    def message_start_event(self) -> MessageStartEvent:
        """The opening event, also used to seed a non-stream projection."""

        return MessageStartEvent(message=self._envelope(content=[]))

    def content_blocks(self) -> list[ContentBlock]:
        """Assemble the finished content blocks in emission order."""

        blocks: list[ContentBlock] = []
        for block in self._order:
            if block.kind == _TEXT:
                blocks.append(
                    TextBlock(
                        text=block.streamed_text,
                        citations=block.citations or None,
                    )
                )
            elif block.kind == _THINKING:
                if not block.streamed_signature:
                    # Reached only if a block escaped ``_close``. Refuse the
                    # whole turn: an unsigned thinking block claims an
                    # integrity guarantee this runtime never made.
                    raise AnthropicAPIError(
                        "a thinking block reached the response envelope "
                        "without an integrity signature",
                        error_type="api_error",
                    )
                blocks.append(
                    ThinkingBlock(
                        thinking=block.streamed_text,
                        signature=block.streamed_signature,
                    )
                )
            elif block.kind == _TOOL_USE:
                blocks.append(
                    ToolUseBlock(
                        id=block.call_id,
                        name=block.name,
                        input=self._decode_tool_input(block),
                    )
                )
            elif block.fixed_content is not None:
                blocks.append(block.fixed_content)
        return blocks

    def terminal_message(self) -> MessagesResponse:
        """The complete non-stream envelope for this turn."""

        if self._failed is not None:
            raise AnthropicAPIError(
                self._failed.message,
                error_type=self._failed.type,
            )
        return self._envelope(content=self.content_blocks())

    # -- lifecycle --------------------------------------------------------

    def _on_turn_started(self) -> tuple[AnthropicStreamEvent, ...]:
        if self._started:
            return ()
        self._started = True
        return (self.message_start_event(),)

    def _on_turn_completed(
        self, event: TurnCompleted
    ) -> tuple[AnthropicStreamEvent, ...]:
        if self._stopped:
            return ()
        if any(not call.completed for call in self._hosted_calls.values()):
            raise AnthropicAPIError(
                "the runtime completed with an open hosted web_fetch call",
                error_type="api_error",
            )
        emitted: list[AnthropicStreamEvent] = []
        emitted.extend(self._close_all_blocks())
        if event.usage is not None:
            self._absorb_usage(event.usage)
        self._state.finish_reason = event.finish_reason
        self._state.stop_reason = self._resolve_stop_reason(event.finish_reason)
        self._state.stop_sequence = event.stop_sequence
        self._stopped = True
        emitted.append(
            MessageDeltaEvent(
                delta=MessageDeltaBody(
                    stop_reason=self._state.stop_reason,
                    stop_sequence=self._state.stop_sequence,
                ),
                usage=self._cumulative_usage(),
            )
        )
        emitted.append(MessageStopEvent())
        return tuple(emitted)

    def _on_turn_failed(
        self, code: str, message: str
    ) -> tuple[AnthropicStreamEvent, ...]:
        if self._stopped:
            return ()
        self._stopped = True
        body = StreamErrorBody(
            type=AnthropicAPIError(message, error_type=code).error_type,
            message=message,
        )
        self._failed = body
        return (StreamErrorEvent(error=body),)

    # -- content parts ----------------------------------------------------

    @staticmethod
    def _content_key(output_index: int, content_index: int) -> tuple[Any, ...]:
        return ("content", output_index, content_index)

    @staticmethod
    def _tool_key(item_id: str, index: int) -> tuple[Any, ...]:
        return ("tool", item_id, index)

    def _on_content_part_started(
        self, event: ContentPartStarted
    ) -> tuple[AnthropicStreamEvent, ...]:
        kind = self._content_kind(event.kind)
        if kind is None:
            return ()
        key = self._content_key(event.output_index, event.content_index)
        _, emitted = self._ensure_content_block(key, kind)
        return emitted

    def _on_text_delta(self, event: TextDelta) -> tuple[AnthropicStreamEvent, ...]:
        key = self._content_key(event.output_index, event.content_index)
        block, emitted = self._ensure_content_block(key, _TEXT)
        if not event.delta:
            return emitted
        block.streamed_text += event.delta
        return (
            *emitted,
            ContentBlockDeltaEvent(
                index=block.index,
                delta=TextDeltaBody(text=event.delta),
            ),
        )

    def _on_reasoning_delta(
        self, event: ReasoningDelta
    ) -> tuple[AnthropicStreamEvent, ...]:
        if not self._thinking.admitted:
            # The runtime reasoned; the client did not ask for Anthropic
            # thinking, or nothing can sign it. Drop the event outright — it
            # is not re-routed into visible text, which would be the same lie
            # wearing a different block type.
            return ()
        key = self._content_key(event.output_index, event.content_index)
        block, emitted = self._ensure_content_block(key, _THINKING)
        if not event.delta:
            return emitted
        block.streamed_text += event.delta
        return (
            *emitted,
            ContentBlockDeltaEvent(
                index=block.index,
                delta=ThinkingDeltaBody(thinking=event.delta),
            ),
        )

    def _on_content_completed(
        self,
        event: TextCompleted | ContentPartCompleted | ReasoningCompleted,
    ) -> tuple[AnthropicStreamEvent, ...]:
        if isinstance(event, ContentPartCompleted):
            kind = self._content_kind(event.kind)
            if kind is None:
                return ()
        elif isinstance(event, ReasoningCompleted):
            if not self._thinking.admitted:
                return ()
            kind = _THINKING
        else:
            kind = _TEXT
        key = self._content_key(event.output_index, event.content_index)
        block, emitted = self._ensure_content_block(key, kind)
        if block.closed:
            return emitted
        return (
            *emitted,
            *self._flush_text_tail(block, event.text),
            *self._close(block),
        )

    def _flush_text_tail(
        self, block: _Block, final_text: str
    ) -> tuple[AnthropicStreamEvent, ...]:
        """Emit only the part of the final text that was never streamed.

        The completion event repeats the whole content; replaying it verbatim
        would duplicate every token the client already accumulated.
        """

        if not final_text.startswith(block.streamed_text):
            # The runtime's authoritative text diverges from the streamed
            # prefix; trusting the prefix would corrupt the accumulated
            # message, so restate the authoritative text as the block content
            # without re-emitting a contradictory delta.
            block.streamed_text = final_text
            return ()
        tail = final_text[len(block.streamed_text) :]
        if not tail:
            return ()
        block.streamed_text = final_text
        body = (
            ThinkingDeltaBody(thinking=tail)
            if block.kind == _THINKING
            else TextDeltaBody(text=tail)
        )
        return (ContentBlockDeltaEvent(index=block.index, delta=body),)

    # -- tool calls -------------------------------------------------------

    def _on_tool_delta(self, event: ToolDelta) -> tuple[AnthropicStreamEvent, ...]:
        key = self._tool_key(event.item_id, event.index)
        block = self._blocks.get(key)
        if block is None:
            block = _Block(index=-1, kind=_TOOL_USE, call_id=event.call_id)
            self._blocks[key] = block
        if event.name and not block.name:
            block.name = event.name
        emitted: list[AnthropicStreamEvent] = []
        if not block.started:
            if not block.name:
                # Anthropic's content_block_start must carry the tool name.
                # Buffer arguments until the runtime names the call rather
                # than emitting a nameless block.
                block.pending_arguments += event.arguments_delta
                return ()
            emitted.extend(self._start_tool_block(block))
        chunk = block.pending_arguments + event.arguments_delta
        block.pending_arguments = ""
        if not chunk:
            return tuple(emitted)
        block.streamed_arguments += chunk
        emitted.append(
            ContentBlockDeltaEvent(
                index=block.index,
                delta=InputJsonDeltaBody(partial_json=chunk),
            )
        )
        return tuple(emitted)

    def _on_tool_completed(
        self,
        *,
        key: tuple[Any, ...],
        call_id: str,
        name: str,
        arguments: str,
    ) -> tuple[AnthropicStreamEvent, ...]:
        block = self._blocks.get(key)
        if block is not None and block.closed:
            # The turn may report the same completed call twice (tool event
            # plus output-item completion). The arguments assemble once.
            return ()
        if block is None:
            block = _Block(index=-1, kind=_TOOL_USE, call_id=call_id)
            self._blocks[key] = block
        block.call_id = block.call_id or call_id
        block.name = name or block.name
        emitted: list[AnthropicStreamEvent] = []
        if not block.started:
            emitted.extend(self._start_tool_block(block))
        pending = block.pending_arguments
        block.pending_arguments = ""
        already_sent = block.streamed_arguments + pending
        # Everything the client has not yet seen, and nothing it already has:
        # the concatenation of every partial_json for this block equals the
        # final arguments exactly once.
        chunk = (
            arguments[len(block.streamed_arguments) :]
            if arguments.startswith(already_sent)
            else ""
        )
        block.streamed_arguments = arguments
        if chunk:
            emitted.append(
                ContentBlockDeltaEvent(
                    index=block.index,
                    delta=InputJsonDeltaBody(partial_json=chunk),
                )
            )
        emitted.extend(self._close(block))
        return tuple(emitted)

    def _start_tool_block(self, block: _Block) -> tuple[AnthropicStreamEvent, ...]:
        block.index = self._next_index
        self._next_index += 1
        block.started = True
        self._order.append(block)
        return (
            ContentBlockStartEvent(
                index=block.index,
                content_block=ToolUseBlock(
                    id=block.call_id,
                    name=block.name,
                    input={},
                ),
            ),
        )

    def _decode_tool_input(self, block: _Block) -> dict[str, Any]:
        raw = block.streamed_arguments.strip()
        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except ValueError as error:
            raise AnthropicAPIError(
                f"tool {block.name!r} produced arguments that are not valid JSON",
                error_type="api_error",
            ) from error
        if not isinstance(decoded, dict):
            raise AnthropicAPIError(
                f"tool {block.name!r} arguments must decode to a JSON object",
                error_type="api_error",
            )
        return decoded

    # -- hosted web fetch -------------------------------------------------

    def _on_hosted_call_started(
        self, event: HostedCallStarted
    ) -> tuple[AnthropicStreamEvent, ...]:
        if event.tool_name != "web_fetch":
            raise AnthropicAPIError(
                f"Anthropic cannot project hosted tool {event.tool_name!r}",
                error_type="api_error",
            )
        if event.call_id in self._hosted_calls:
            raise AnthropicAPIError(
                f"hosted call {event.call_id!r} started more than once",
                error_type="api_error",
            )
        action = dict(event.action)
        tool_use_id = _server_tool_use_id(event.call_id)
        self._hosted_calls[event.call_id] = _HostedCall(
            call_id=event.call_id,
            item_id=event.item_id,
            tool_use_id=tool_use_id,
            action=action,
        )
        payload = json.dumps(
            action,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        block = _Block(
            index=self._next_index,
            kind=_SERVER_TOOL_USE,
            call_id=tool_use_id,
            name="web_fetch",
            streamed_arguments=payload,
            started=True,
            closed=True,
            fixed_content=ServerToolUseBlock(
                id=tool_use_id,
                name="web_fetch",
                input=action,
            ),
        )
        self._next_index += 1
        self._order.append(block)
        return (
            ContentBlockStartEvent(
                index=block.index,
                content_block=ServerToolUseBlock(
                    id=tool_use_id,
                    name="web_fetch",
                    input={},
                ),
            ),
            ContentBlockDeltaEvent(
                index=block.index,
                delta=InputJsonDeltaBody(partial_json=payload),
            ),
            ContentBlockStopEvent(index=block.index),
        )

    def _on_hosted_call_result(
        self, event: HostedCallResult
    ) -> tuple[AnthropicStreamEvent, ...]:
        call = self._require_hosted_call(event.call_id, event.item_id, event.tool_name)
        if call.result is not None:
            raise AnthropicAPIError(
                f"hosted call {event.call_id!r} produced more than one result",
                error_type="api_error",
            )
        if event.result.get("kind") != "document":
            raise AnthropicAPIError(
                "Anthropic web_fetch cannot project a PDF or search-result payload",
                error_type="api_error",
            )
        call.result = event.result
        return ()

    def _on_hosted_call_completed(
        self, event: HostedCallCompleted
    ) -> tuple[AnthropicStreamEvent, ...]:
        call = self._require_hosted_call(event.call_id, event.item_id, event.tool_name)
        if call.completed:
            raise AnthropicAPIError(
                f"hosted call {event.call_id!r} completed more than once",
                error_type="api_error",
            )
        call.completed = True
        if event.status == "completed":
            if call.result is None:
                raise AnthropicAPIError(
                    f"hosted call {event.call_id!r} completed without a result",
                    error_type="api_error",
                )
            if event.receipt.get("final_url") != call.result.get(
                "url"
            ) or event.receipt.get("result_digest") != call.result.get("digest"):
                raise AnthropicAPIError(
                    f"hosted call {event.call_id!r} result contradicts its receipt",
                    error_type="api_error",
                )
            content = self._web_fetch_success(call)
            call.document_index = self._document_count
            self._document_count += 1
        else:
            if call.result is not None:
                raise AnthropicAPIError(
                    f"failed hosted call {event.call_id!r} carried a success result",
                    error_type="api_error",
                )
            content = WebFetchToolResultBlock(
                tool_use_id=call.tool_use_id,
                content=WebFetchToolResultErrorBlock(
                    error_code=_web_fetch_error_code(event.receipt),
                ),
            )
        return self._emit_whole_hosted_result(call, content)

    def _web_fetch_success(self, call: _HostedCall) -> WebFetchToolResultBlock:
        result = call.result
        if result is None:
            raise AssertionError("success projection requires a stored result")
        url = result.get("url")
        content = result.get("content")
        media_type = result.get("media_type")
        retrieved_at = result.get("retrieved_at")
        if not isinstance(url, str) or not url:
            raise AnthropicAPIError(
                "web_fetch result has no final URL", error_type="api_error"
            )
        if not isinstance(content, str):
            raise AnthropicAPIError(
                "web_fetch result has no bounded text", error_type="api_error"
            )
        if not isinstance(media_type, str) or not _is_text_media_type(media_type):
            raise AnthropicAPIError(
                f"web_fetch result media_type {media_type!r} is not text-family",
                error_type="api_error",
            )
        if isinstance(retrieved_at, bool) or not isinstance(retrieved_at, int):
            raise AnthropicAPIError(
                "web_fetch result has no integer retrieval time",
                error_type="api_error",
            )
        timestamp = (
            datetime.fromtimestamp(retrieved_at, UTC).isoformat().replace("+00:00", "Z")
        )
        return WebFetchToolResultBlock(
            tool_use_id=call.tool_use_id,
            content=WebFetchResultBlock(
                url=url,
                retrieved_at=timestamp,
                content=DocumentBlock(
                    source=PlainTextSource(data=content),
                    title=url,
                ),
            ),
        )

    def _emit_whole_hosted_result(
        self,
        call: _HostedCall,
        content: WebFetchToolResultBlock,
    ) -> tuple[AnthropicStreamEvent, ...]:
        if call.result_emitted:
            raise AnthropicAPIError(
                f"hosted call {call.call_id!r} projected more than one result",
                error_type="api_error",
            )
        call.result_emitted = True
        block = _Block(
            index=self._next_index,
            kind=_WEB_FETCH_RESULT,
            started=True,
            closed=True,
            fixed_content=content,
        )
        self._next_index += 1
        self._order.append(block)
        return (
            ContentBlockStartEvent(index=block.index, content_block=content),
            ContentBlockStopEvent(index=block.index),
        )

    def _on_hosted_citation(
        self, event: HostedCitation
    ) -> tuple[AnthropicStreamEvent, ...]:
        call = self._hosted_calls.get(event.source_call_id)
        if (
            call is None
            or not call.completed
            or not call.result_emitted
            or call.document_index is None
            or call.result is None
        ):
            raise AnthropicAPIError(
                "hosted citation names no completed web_fetch receipt",
                error_type="api_error",
            )
        source_url = call.result.get("url")
        source_text = call.result.get("content")
        if source_url != event.source_url or not isinstance(source_text, str):
            raise AnthropicAPIError(
                "hosted citation contradicts its web_fetch result URL",
                error_type="api_error",
            )
        if event.source_end > len(source_text):
            raise AnthropicAPIError(
                "hosted citation source range exceeds its web_fetch document",
                error_type="api_error",
            )
        key = self._content_key(event.output_index, event.content_index)
        block = self._blocks.get(key)
        if block is None or block.kind != _TEXT:
            raise AnthropicAPIError(
                "hosted citation names no active continuation text block",
                error_type="api_error",
            )
        if (
            event.output_end > len(block.streamed_text)
            or block.streamed_text[event.output_start : event.output_end]
            != event.cited_text
        ):
            raise AnthropicAPIError(
                "hosted citation output range is not verbatim continuation text",
                error_type="api_error",
            )
        citation = CitationCharLocation(
            cited_text=event.cited_text,
            document_index=call.document_index,
            document_title=event.source_url,
            start_char_index=event.source_start,
            end_char_index=event.source_end,
        )
        if citation in block.citations:
            raise AnthropicAPIError(
                "hosted citation was projected more than once",
                error_type="api_error",
            )
        block.citations.append(citation)
        return (
            ContentBlockDeltaEvent(
                index=block.index,
                delta=CitationsDeltaBody(citation=citation),
            ),
        )

    def _require_hosted_call(
        self,
        call_id: str,
        item_id: str,
        tool_name: str,
    ) -> _HostedCall:
        call = self._hosted_calls.get(call_id)
        if call is None or call.item_id != item_id or tool_name != "web_fetch":
            raise AnthropicAPIError(
                f"hosted event does not match web_fetch call {call_id!r}",
                error_type="api_error",
            )
        return call

    # -- output items -----------------------------------------------------

    def _on_output_item_completed(
        self, event: OutputItemCompleted
    ) -> tuple[AnthropicStreamEvent, ...]:
        if event.kind != "function_call":
            return ()
        if event.call_id is None or event.name is None or event.arguments is None:
            return ()
        return self._on_tool_completed(
            key=self._tool_key(event.item_id, event.index),
            call_id=event.call_id,
            name=event.name,
            arguments=event.arguments,
        )

    # -- shared block plumbing -------------------------------------------

    def _content_kind(self, kind: str) -> str | None:
        if kind == TEXT_CONTENT_KIND:
            return _TEXT
        if kind == REASONING_CONTENT_KIND:
            # Unadmitted reasoning has no Anthropic block kind at all, so no
            # content_block_start is ever opened for it.
            return _THINKING if self._thinking.admitted else None
        return None

    def _ensure_content_block(
        self, key: tuple[Any, ...], kind: str
    ) -> tuple[_Block, tuple[AnthropicStreamEvent, ...]]:
        block = self._blocks.get(key)
        if block is not None:
            return block, ()
        block = _Block(index=self._next_index, kind=kind, started=True)
        self._next_index += 1
        self._blocks[key] = block
        self._order.append(block)
        content: ContentBlock = (
            # The opening block carries no signature yet — the protocol
            # delivers it later as a signature_delta. Saying ``""`` here is
            # explicit, and only reachable once thinking has been admitted.
            ThinkingBlock(thinking="", signature="")
            if kind == _THINKING
            else TextBlock(text="")
        )
        return block, (
            ContentBlockStartEvent(index=block.index, content_block=content),
        )

    def _close(self, block: _Block) -> tuple[AnthropicStreamEvent, ...]:
        if block.closed or not block.started:
            return ()
        block.closed = True
        emitted: list[AnthropicStreamEvent] = []
        if block.kind == _THINKING:
            # Signing happens at close, on the assembled text, and it either
            # produces a real signature or raises. There is no branch here
            # that closes a thinking block unsigned.
            signature = self._thinking.sign(
                message_id=self._message_id,
                index=block.index,
                thinking=block.streamed_text,
            )
            block.streamed_signature = signature.value
            emitted.append(
                ContentBlockDeltaEvent(
                    index=block.index,
                    delta=SignatureDeltaBody(signature=signature.value),
                )
            )
        emitted.append(ContentBlockStopEvent(index=block.index))
        return tuple(emitted)

    def _close_all_blocks(self) -> tuple[AnthropicStreamEvent, ...]:
        emitted: list[AnthropicStreamEvent] = []
        for block in self._order:
            emitted.extend(self._close(block))
        return tuple(emitted)

    # -- usage and finality ----------------------------------------------

    def _stamp_tier(self, usage: Usage) -> Usage:
        """Record the lane that served the turn on every usage snapshot."""

        return usage.model_copy(update={"service_tier": self._service_tier})

    def _absorb_usage(self, event: UsageUpdate) -> None:
        self._state.usage = Usage(
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            cache_creation_input_tokens=(
                event.cache_write_input_tokens
                if event.cache_write_input_tokens
                else None
            ),
            cache_read_input_tokens=(
                event.cached_input_tokens if event.cached_input_tokens else None
            ),
            service_tier=self._service_tier,
        )

    def _cumulative_usage(self) -> MessageDeltaUsage:
        usage = self._state.usage
        return MessageDeltaUsage(
            output_tokens=usage.output_tokens,
            input_tokens=usage.input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
        )

    def _resolve_stop_reason(  # noqa: PLR0911 - protocol mapping table
        self, finish_reason: str | None
    ) -> StopReason:
        normalized = (finish_reason or "").strip().lower()
        if normalized in _TRUNCATION_REASONS:
            return StopReason.MAX_TOKENS
        if normalized in _TOOL_REASONS:
            return StopReason.TOOL_USE
        if normalized in _STOP_SEQUENCE_REASONS:
            return StopReason.STOP_SEQUENCE
        if normalized in _REFUSAL_REASONS:
            return StopReason.REFUSAL
        if normalized in _PAUSE_REASONS:
            return StopReason.PAUSE_TURN
        if any(block.kind == _TOOL_USE for block in self._order):
            return StopReason.TOOL_USE
        return StopReason.END_TURN

    def _envelope(self, *, content: list[ContentBlock]) -> MessagesResponse:
        return MessagesResponse(
            id=self._message_id,
            content=content,
            model=self._model_alias,
            stop_reason=self._state.stop_reason,
            stop_sequence=self._state.stop_sequence,
            usage=self._state.usage,
        )


def _server_tool_use_id(call_id: str) -> str:
    digest = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24]
    return f"srvtoolu_{digest}"


def _is_text_media_type(media_type: str) -> bool:
    normalized = media_type.partition(";")[0].strip().lower()
    return normalized.startswith("text/") or normalized in {
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }


def _web_fetch_error_code(receipt: Mapping[str, Any]) -> WebFetchErrorCode:
    error = receipt.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    if not isinstance(code, str) or code not in HOSTED_ERROR_CODES:
        raise AnthropicAPIError(
            "failed web_fetch receipt has no registered hosted error code",
            error_type="api_error",
        )
    mapped: WebFetchErrorCode
    if code == "tool_round_limit":
        mapped = "max_uses_exceeded"
    elif code == "fetch_url_fetch_status" and receipt.get("http_status") == 429:
        mapped = "too_many_requests"
    elif code in {
        "fetch_unsupported_media_type",
        "fetch_missing_media_type",
        "fetch_invalid_media_type",
        "fetch_invalid_fetch_media_types",
    }:
        mapped = "unsupported_content_type"
    elif code in {
        "invalid_tool_arguments",
        "invalid_tool_result",
        "tool_not_allowed",
        "fetch_invalid_fetch_budget",
        "fetch_token_budget",
    }:
        mapped = "invalid_tool_input"
    elif code == "tool_arguments_too_large":
        mapped = "url_too_long"
    elif code in {
        "fetch_invalid_url",
        "fetch_invalid_url_scheme",
        "fetch_url_credentials_forbidden",
        "fetch_url_target_blocked",
        "fetch_redirect_not_allowed",
        "fetch_url_not_allowed",
    }:
        mapped = "url_not_allowed"
    elif code in {
        "fetch_dns_resolution_failed",
        "fetch_redirect_limit_exceeded",
        "fetch_invalid_redirect",
        "fetch_url_fetch_status",
        "fetch_url_fetch_timeout",
        "fetch_url_fetch_failed",
        "fetch_url_fetch_cancelled",
    }:
        mapped = "url_not_accessible"
    else:
        mapped = "unavailable"
    return mapped


__all__ = [
    "AnthropicMessageProjector",
    "ProjectionState",
    "ThinkingProjection",
    "ThinkingSignature",
    "ThinkingSignatureOwner",
]
