"""Qwen4 output-to-turn translation for the fused runtime.

The tensor owner supplies already separated reasoning text and raw assistant
text. Raw assistant text is always passed through the target-owned
``IncrementalToolParser`` before any visible text or tool event is emitted.
This module owns neither model stepping nor protocol projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...tools.dialects.qwen import QwenToolDialect
from ...tools.parser import (
    IncrementalToolParser,
    ParsedToolCall,
    ParsedToolDelta,
)
from ..contracts import GenerationRequest
from ..events import (
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnEvent,
    UsageUpdate,
)


class Qwen4OutputError(RuntimeError):
    """Qwen4 output cannot be represented without changing its meaning."""


@dataclass(frozen=True, slots=True)
class Qwen4OutputChunk:
    """One observed decoder chunk, before target event sequencing."""

    text_delta: str = ""
    reasoning_delta: str = ""
    usage: UsageUpdate | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text_delta, str):
            raise TypeError("text_delta must be a string")
        if not isinstance(self.reasoning_delta, str):
            raise TypeError("reasoning_delta must be a string")
        if self.usage is not None and not isinstance(self.usage, UsageUpdate):
            raise TypeError("usage must be a UsageUpdate")


@runtime_checkable
class Qwen4OutputEncoderPort(Protocol):
    def feed(self, chunk: Qwen4OutputChunk) -> tuple[TurnEvent, ...]: ...

    def finish(
        self,
        final_usage: UsageUpdate | None = None,
        *,
        finish_reason: str = "stop",
    ) -> tuple[TurnEvent, ...]: ...


@runtime_checkable
class Qwen4OutputEncoderFactoryPort(Protocol):
    def create(self, request: GenerationRequest) -> Qwen4OutputEncoderPort: ...


@dataclass(slots=True)
class _ContentState:
    kind: str
    output_index: int
    item_id: str
    text: str = ""


@dataclass(slots=True)
class _ToolState:
    parser_index: int
    parser_call_id: str
    output_index: int
    item_id: str
    call_id: str
    name: str
    arguments: str = ""


class Qwen4TurnEventEncoder:
    """Build a complete canonical output lifecycle for one Qwen4 response."""

    def __init__(
        self,
        response_id: str,
        *,
        parser: IncrementalToolParser | None = None,
    ) -> None:
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must not be empty")
        if parser is not None and not isinstance(parser, IncrementalToolParser):
            raise TypeError("parser must be an IncrementalToolParser")
        self._response_id = response_id
        self._parser = parser or IncrementalToolParser(QwenToolDialect())
        self._next_output_index = 0
        self._reasoning: _ContentState | None = None
        self._text: _ContentState | None = None
        self._tools: dict[str, _ToolState] = {}
        self._tool_ids_by_parser_index: dict[int, str] = {}
        self._reasoning_text = ""
        self._visible_text = ""
        self._usage: UsageUpdate | None = None
        self._ordinary_output_started = False
        self._finished = False

    @property
    def response_id(self) -> str:
        return self._response_id

    @property
    def finished(self) -> bool:
        return self._finished

    def feed(self, chunk: Qwen4OutputChunk) -> tuple[TurnEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Qwen4 output encoder")
        if not isinstance(chunk, Qwen4OutputChunk):
            raise TypeError("encoder input must be a Qwen4OutputChunk")
        if chunk.reasoning_delta and self._ordinary_output_started:
            raise Qwen4OutputError("reasoning cannot resume after assistant output")
        if (
            chunk.reasoning_delta
            and chunk.text_delta
            and chunk.reasoning_delta == chunk.text_delta
        ):
            raise Qwen4OutputError(
                "the same decoder bytes cannot be reasoning and visible output"
            )
        self._validate_usage(chunk.usage)

        visible_delta = ""
        tool_deltas: tuple[ParsedToolDelta, ...] = ()
        if chunk.text_delta:
            try:
                visible_delta, tool_deltas = self._parser.feed(chunk.text_delta)
            except ValueError as error:
                raise Qwen4OutputError(f"Qwen tool output rejected: {error}") from error

        events: list[TurnEvent] = []
        if chunk.reasoning_delta:
            events.extend(self._append_reasoning(chunk.reasoning_delta))
        if visible_delta or tool_deltas:
            events.extend(self._close_reasoning())
            self._ordinary_output_started = True
        if visible_delta:
            events.extend(self._append_text(visible_delta))
        if tool_deltas:
            events.extend(self._close_text())
            for delta in tool_deltas:
                events.extend(self._append_tool(delta))
        events.extend(self._accept_usage(chunk.usage))
        return tuple(events)

    def finish(
        self,
        final_usage: UsageUpdate | None = None,
        *,
        finish_reason: str = "stop",
    ) -> tuple[TurnEvent, ...]:
        if self._finished:
            raise RuntimeError("Qwen4 output encoder is already finished")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            raise ValueError("finish_reason must not be empty")
        self._validate_usage(final_usage)
        try:
            visible_delta, calls = self._parser.finish()
        except ValueError as error:
            raise Qwen4OutputError(f"Qwen tool output rejected: {error}") from error

        events: list[TurnEvent] = []
        if visible_delta or calls:
            events.extend(self._close_reasoning())
            self._ordinary_output_started = True
        if visible_delta:
            events.extend(self._append_text(visible_delta))

        if (
            self._reasoning_text
            and self._visible_text
            and self._reasoning_text == self._visible_text
        ):
            raise Qwen4OutputError(
                "reasoning and visible output cannot contain the same full text"
            )

        terminal_item_status = (
            "incomplete" if finish_reason == "length" else "completed"
        )
        events.extend(self._close_reasoning(status=terminal_item_status))
        events.extend(self._close_text(status=terminal_item_status))
        events.extend(self._complete_tools(calls, status=terminal_item_status))
        events.extend(self._accept_usage(final_usage))
        self._finished = True
        return tuple(events)

    def _append_reasoning(self, delta: str) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        if self._reasoning is None:
            self._reasoning = self._new_content("reasoning")
            events.extend(self._start_content(self._reasoning))
        state = self._reasoning
        state.text += delta
        self._reasoning_text += delta
        events.append(
            ReasoningDelta(
                delta=delta,
                item_id=state.item_id,
                output_index=state.output_index,
                content_index=0,
            )
        )
        return events

    def _append_text(self, delta: str) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        if self._text is None:
            self._text = self._new_content("message")
            events.extend(self._start_content(self._text))
        state = self._text
        state.text += delta
        self._visible_text += delta
        events.append(
            TextDelta(
                delta=delta,
                item_id=state.item_id,
                output_index=state.output_index,
                content_index=0,
            )
        )
        return events

    def _append_tool(self, delta: ParsedToolDelta) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        indexed_call_id = self._tool_ids_by_parser_index.get(delta.index)
        if indexed_call_id is not None and indexed_call_id != delta.call_id:
            raise Qwen4OutputError("tool parser changed call identity for an index")

        state = self._tools.get(delta.call_id)
        if state is None:
            if delta.name is None:
                raise Qwen4OutputError("first tool delta must include its name")
            output_index = self._claim_output_index()
            state = _ToolState(
                parser_index=delta.index,
                parser_call_id=delta.call_id,
                output_index=output_index,
                item_id=self._item_id("function_call", output_index),
                call_id=f"{self._response_id}:{delta.call_id}",
                name=delta.name,
            )
            self._tools[delta.call_id] = state
            self._tool_ids_by_parser_index[delta.index] = delta.call_id
            events.append(
                OutputItemStarted(
                    kind="function_call",
                    index=state.output_index,
                    item_id=state.item_id,
                )
            )
        elif state.parser_index != delta.index:
            raise Qwen4OutputError("tool parser changed the index for a call")
        elif delta.name is not None and delta.name != state.name:
            raise Qwen4OutputError("tool parser changed the tool name")

        state.arguments += delta.arguments_delta
        events.append(
            ToolDelta(
                index=state.output_index,
                call_id=state.call_id,
                item_id=state.item_id,
                name=delta.name,
                arguments_delta=delta.arguments_delta,
            )
        )
        return events

    def _new_content(self, kind: str) -> _ContentState:
        output_index = self._claim_output_index()
        return _ContentState(
            kind=kind,
            output_index=output_index,
            item_id=self._item_id(kind, output_index),
        )

    @staticmethod
    def _start_content(state: _ContentState) -> list[TurnEvent]:
        content_kind = (
            "reasoning_summary_text" if state.kind == "reasoning" else "output_text"
        )
        return [
            OutputItemStarted(state.kind, state.output_index, state.item_id),
            ContentPartStarted(content_kind, state.output_index, 0, state.item_id),
        ]

    def _close_reasoning(self, *, status: str = "completed") -> list[TurnEvent]:
        state = self._reasoning
        if state is None:
            return []
        self._reasoning = None
        return [
            ReasoningCompleted(state.text, state.item_id, state.output_index, 0),
            ContentPartCompleted(
                "reasoning_summary_text",
                state.output_index,
                0,
                state.item_id,
                state.text,
            ),
            OutputItemCompleted(
                "reasoning",
                state.output_index,
                state.item_id,
                text=state.text,
                status=status,
            ),
        ]

    def _close_text(self, *, status: str = "completed") -> list[TurnEvent]:
        state = self._text
        if state is None:
            return []
        self._text = None
        return [
            TextCompleted(state.text, state.item_id, state.output_index, 0),
            ContentPartCompleted(
                "output_text", state.output_index, 0, state.item_id, state.text
            ),
            OutputItemCompleted(
                "message",
                state.output_index,
                state.item_id,
                text=state.text,
                status=status,
            ),
        ]

    def _complete_tools(
        self,
        calls: tuple[ParsedToolCall, ...],
        *,
        status: str = "completed",
    ) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        if len(calls) != len(self._tools):
            raise Qwen4OutputError("final tool calls do not match streamed tool calls")
        for call in sorted(calls, key=lambda item: item.index):
            state = self._tools.get(call.call_id)
            if state is None or state.parser_index != call.index:
                raise Qwen4OutputError("final tool call identity was not streamed")
            if state.name != call.name or state.arguments != call.arguments:
                raise Qwen4OutputError(
                    "final tool call does not match streamed name and arguments"
                )
            events.extend(
                (
                    ToolCompleted(
                        index=state.output_index,
                        call_id=state.call_id,
                        item_id=state.item_id,
                        name=state.name,
                        arguments=state.arguments,
                    ),
                    OutputItemCompleted(
                        "function_call",
                        state.output_index,
                        state.item_id,
                        call_id=state.call_id,
                        name=state.name,
                        arguments=state.arguments,
                        status=status,
                    ),
                )
            )
        return events

    def _validate_usage(self, usage: UsageUpdate | None) -> None:
        if usage is None or self._usage is None:
            return
        previous = self._usage
        if (
            usage.input_tokens < previous.input_tokens
            or usage.output_tokens < previous.output_tokens
            or usage.total_tokens < previous.total_tokens
            or usage.cached_input_tokens < previous.cached_input_tokens
            or usage.cache_write_input_tokens < previous.cache_write_input_tokens
            or usage.reasoning_output_tokens < previous.reasoning_output_tokens
        ):
            raise Qwen4OutputError(
                "usage observations must be cumulative and monotonic"
            )

    def _accept_usage(self, usage: UsageUpdate | None) -> list[TurnEvent]:
        if usage is None or usage == self._usage:
            return []
        self._usage = usage
        return [usage]

    def _claim_output_index(self) -> int:
        output_index = self._next_output_index
        self._next_output_index += 1
        return output_index

    def _item_id(self, kind: str, output_index: int) -> str:
        return f"{self._response_id}:{kind}:{output_index}"


class Qwen4TurnEventEncoderFactory:
    """Create one parser and one event encoder per generation response."""

    def create(self, request: GenerationRequest) -> Qwen4TurnEventEncoder:
        if not isinstance(request, GenerationRequest):
            raise TypeError("encoder factory requires a GenerationRequest")
        return Qwen4TurnEventEncoder(request.response_id)


__all__ = [
    "Qwen4OutputChunk",
    "Qwen4OutputEncoderFactoryPort",
    "Qwen4OutputEncoderPort",
    "Qwen4OutputError",
    "Qwen4TurnEventEncoder",
    "Qwen4TurnEventEncoderFactory",
]
