"""Project typed runtime turn events onto the Anthropic Messages protocol.

This is the only place that knows how a generation turn becomes Anthropic
wire shapes. It owns content-block indices, cumulative usage and stop-reason
mapping for *both* transports, so the streaming lifecycle and the non-stream
envelope can never disagree about what the model produced.

The projector consumes ``mlx_batch_server.runtime.events`` — the shared,
protocol-neutral event family — and never reaches into another protocol's
internals. Provider semantics are mapped, not flattened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mlx_batch_server.runtime.events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartCompleted,
    ContentPartStarted,
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

from .anthropic_schema import (
    AnthropicStreamEvent,
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJsonDeltaBody,
    MessageDeltaBody,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessagesResponse,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
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
)
from .errors import AnthropicAPIError

_TEXT = "text"
_THINKING = "thinking"
_TOOL_USE = "tool_use"

#: Runtime finish reasons that mean the output was truncated. Truncation is
#: resolved before tool use: a tool call cut off mid-arguments is a
#: ``max_tokens`` turn, not a ``tool_use`` turn.
_TRUNCATION_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})
_TOOL_REASONS = frozenset({"tool_calls", "tool_use", "function_call"})
_REFUSAL_REASONS = frozenset({"refusal", "content_filter"})
_PAUSE_REASONS = frozenset({"pause_turn", "pause"})
_STOP_SEQUENCE_REASONS = frozenset({"stop_sequence", "stop_sequences"})


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
    ) -> None:
        if not message_id.strip():
            raise ValueError("message_id must not be empty")
        if not model_alias.strip():
            raise ValueError("model_alias must not be empty")
        self._message_id = message_id
        # The public alias the caller asked for. The runtime's resolved
        # physical model identity is deliberately never substituted here: the
        # ``model`` a client sees at message_start is the same one it sees in
        # the terminal envelope.
        self._model_alias = model_alias
        self._state = ProjectionState(usage=initial_usage or Usage())
        self._blocks: dict[tuple[Any, ...], _Block] = {}
        self._order: list[_Block] = []
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

    def observe(self, event: TurnEvent) -> tuple[AnthropicStreamEvent, ...]:
        """Project one runtime event onto zero or more Anthropic events."""

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
                blocks.append(TextBlock(text=block.streamed_text))
            elif block.kind == _THINKING:
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
        emitted: list[AnthropicStreamEvent] = []
        emitted.extend(self._close_all_blocks())
        if event.usage is not None:
            self._absorb_usage(event.usage)
        self._state.finish_reason = event.finish_reason
        self._state.stop_reason = self._resolve_stop_reason(event.finish_reason)
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

    @staticmethod
    def _content_kind(kind: str) -> str | None:
        if kind == TEXT_CONTENT_KIND:
            return _TEXT
        if kind == REASONING_CONTENT_KIND:
            return _THINKING
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
            ThinkingBlock(thinking="") if kind == _THINKING else TextBlock(text="")
        )
        return block, (
            ContentBlockStartEvent(index=block.index, content_block=content),
        )

    def _close(self, block: _Block) -> tuple[AnthropicStreamEvent, ...]:
        if block.closed or not block.started:
            return ()
        block.closed = True
        emitted: list[AnthropicStreamEvent] = []
        if block.kind == _THINKING and block.streamed_signature:
            emitted.append(
                ContentBlockDeltaEvent(
                    index=block.index,
                    delta=SignatureDeltaBody(signature=block.streamed_signature),
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
        )

    def _cumulative_usage(self) -> MessageDeltaUsage:
        usage = self._state.usage
        return MessageDeltaUsage(
            output_tokens=usage.output_tokens,
            input_tokens=usage.input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
        )

    def _resolve_stop_reason(self, finish_reason: str | None) -> StopReason:
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


__all__ = ["AnthropicMessageProjector", "ProjectionState"]
