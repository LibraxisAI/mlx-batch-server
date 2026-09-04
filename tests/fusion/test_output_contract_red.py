"""RED contracts for Qwen4 output encoding through the canonical tool parser.

These tests are intentionally not executed while Compile Embargo is HOLD.
"""

from __future__ import annotations

import pytest

from mlx_batch_server.runtime.events import (
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
    UsageUpdate,
)
from mlx_batch_server.runtime.fusion.output import (
    Qwen4OutputChunk,
    Qwen4OutputError,
    Qwen4TurnEventEncoder,
)
from mlx_batch_server.tools.parser import (
    DialectParse,
    IncrementalToolParser,
)


def _all_events(
    encoder: Qwen4TurnEventEncoder,
    chunks: tuple[Qwen4OutputChunk, ...],
    *,
    final_usage: UsageUpdate | None = None,
    finish_reason: str = "stop",
) -> tuple[object, ...]:
    events: list[object] = []
    for chunk in chunks:
        events.extend(encoder.feed(chunk))
    events.extend(encoder.finish(final_usage, finish_reason=finish_reason))
    return tuple(events)


def test_qwen_reasoning_text_and_tool_call_form_one_complete_turn_lifecycle() -> None:
    encoder = Qwen4TurnEventEncoder("resp_qwen")
    usage = UsageUpdate(
        input_tokens=12,
        output_tokens=5,
        total_tokens=17,
        reasoning_output_tokens=2,
    )

    events = _all_events(
        encoder,
        (
            Qwen4OutputChunk(reasoning_delta="inspect evidence"),
            Qwen4OutputChunk(text_delta="Answer: "),
            Qwen4OutputChunk(
                text_delta='<tool_call>{"name":"lookup","arguments":{"id":'
            ),
            Qwen4OutputChunk(text_delta='"LBRX-42"}}</tool_call>'),
        ),
        final_usage=usage,
    )

    assert [type(event) for event in events] == [
        OutputItemStarted,
        ContentPartStarted,
        ReasoningDelta,
        ReasoningCompleted,
        ContentPartCompleted,
        OutputItemCompleted,
        OutputItemStarted,
        ContentPartStarted,
        TextDelta,
        TextCompleted,
        ContentPartCompleted,
        OutputItemCompleted,
        OutputItemStarted,
        ToolDelta,
        ToolDelta,
        ToolCompleted,
        OutputItemCompleted,
        UsageUpdate,
    ]
    assert events[0] == OutputItemStarted("reasoning", 0, "resp_qwen:reasoning:0")
    assert events[6] == OutputItemStarted("message", 1, "resp_qwen:message:1")
    assert events[12] == OutputItemStarted(
        "function_call",
        2,
        "resp_qwen:function_call:2",
    )
    tool_deltas = tuple(event for event in events if isinstance(event, ToolDelta))
    assert tool_deltas[0].name == "lookup"
    assert tool_deltas[0].call_id == "resp_qwen:call_qwen_0"
    assert "".join(event.arguments_delta for event in tool_deltas) == (
        '{"id":"LBRX-42"}'
    )
    completed = next(event for event in events if isinstance(event, ToolCompleted))
    assert completed.arguments == '{"id":"LBRX-42"}'
    assert completed.call_id == "resp_qwen:call_qwen_0"
    assert "".join(event.delta for event in events if isinstance(event, TextDelta)) == (
        "Answer: "
    )
    assert all(
        "<tool_call>" not in event.delta
        for event in events
        if isinstance(event, TextDelta)
    )
    assert events[-1] == usage
    assert encoder.finished is True


def test_every_raw_text_chunk_passes_through_incremental_tool_parser() -> None:
    class RecordingDialect:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def parse(self, text: str, *, final: bool) -> DialectParse:
            self.calls.append((text, final))
            return DialectParse(visible_text=text.upper())

    dialect = RecordingDialect()
    parser = IncrementalToolParser(dialect)
    encoder = Qwen4TurnEventEncoder("resp_parser", parser=parser)

    events = _all_events(
        encoder,
        (Qwen4OutputChunk(text_delta="ab"), Qwen4OutputChunk(text_delta="cd")),
    )

    assert dialect.calls == [("ab", False), ("abcd", False), ("abcd", True)]
    assert "".join(event.delta for event in events if isinstance(event, TextDelta)) == (
        "ABCD"
    )
    done = next(event for event in events if isinstance(event, TextCompleted))
    assert done.text == "ABCD"


def test_usage_is_cumulative_monotonic_and_duplicate_snapshots_are_suppressed() -> None:
    encoder = Qwen4TurnEventEncoder("resp_usage")
    first = UsageUpdate(10, 1, 11, cached_input_tokens=4)
    final = UsageUpdate(
        10,
        3,
        13,
        cached_input_tokens=4,
        reasoning_output_tokens=1,
    )

    assert encoder.feed(Qwen4OutputChunk(usage=first)) == (first,)
    assert encoder.feed(Qwen4OutputChunk(usage=first)) == ()
    assert encoder.finish(final) == (final,)

    regressing = Qwen4TurnEventEncoder("resp_regressing")
    regressing.feed(Qwen4OutputChunk(usage=final))
    with pytest.raises(Qwen4OutputError, match="monotonic"):
        regressing.feed(Qwen4OutputChunk(usage=first))


def test_length_finish_marks_only_the_terminal_open_item_incomplete() -> None:
    encoder = Qwen4TurnEventEncoder("resp_limited")
    events = list(encoder.feed(Qwen4OutputChunk(reasoning_delta="finished thought")))
    events.extend(encoder.feed(Qwen4OutputChunk(text_delta="truncated answer")))

    events.extend(encoder.finish(finish_reason="length"))
    completed_items = tuple(
        event for event in events if isinstance(event, OutputItemCompleted)
    )

    assert tuple((event.kind, event.status) for event in completed_items) == (
        ("reasoning", "completed"),
        ("message", "incomplete"),
    )


def test_output_item_completion_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        OutputItemCompleted(
            "message",
            0,
            "msg_bad",
            text="bad",
            status="in_progress",
        )


def test_reasoning_cannot_resume_or_duplicate_visible_output() -> None:
    same_chunk = Qwen4TurnEventEncoder("resp_same_chunk")
    with pytest.raises(Qwen4OutputError, match="same decoder bytes"):
        same_chunk.feed(Qwen4OutputChunk(text_delta="same", reasoning_delta="same"))

    resumed = Qwen4TurnEventEncoder("resp_resumed")
    resumed.feed(Qwen4OutputChunk(text_delta="answer"))
    with pytest.raises(Qwen4OutputError, match="cannot resume"):
        resumed.feed(Qwen4OutputChunk(reasoning_delta="late thought"))

    duplicated = Qwen4TurnEventEncoder("resp_duplicated")
    duplicated.feed(Qwen4OutputChunk(reasoning_delta="same"))
    duplicated.feed(Qwen4OutputChunk(text_delta="same"))
    with pytest.raises(Qwen4OutputError, match="same full text"):
        duplicated.finish()


def test_multiple_tool_calls_keep_stable_parser_and_response_scoped_identity() -> None:
    encoder = Qwen4TurnEventEncoder("resp_tools")
    events = _all_events(
        encoder,
        (
            Qwen4OutputChunk(
                text_delta=(
                    '<tool_call>{"name":"first","arguments":{}}</tool_call>'
                    '<tool_call>{"name":"second","arguments":{"n":2}}</tool_call>'
                )
            ),
        ),
    )

    started = tuple(
        event
        for event in events
        if isinstance(event, OutputItemStarted) and event.kind == "function_call"
    )
    completed = tuple(event for event in events if isinstance(event, ToolCompleted))
    assert tuple(event.index for event in started) == (0, 1)
    assert tuple(event.call_id for event in completed) == (
        "resp_tools:call_qwen_0",
        "resp_tools:call_qwen_1",
    )
    assert tuple(event.name for event in completed) == ("first", "second")


def test_malformed_qwen_tool_envelope_fails_closed_without_visible_marker() -> None:
    encoder = Qwen4TurnEventEncoder("resp_malformed")

    emitted = encoder.feed(
        Qwen4OutputChunk(text_delta='<tool_call>{"name":"lookup","arguments":{')
    )
    assert all(
        "<tool_call>" not in event.delta
        for event in emitted
        if isinstance(event, TextDelta)
    )
    with pytest.raises(Qwen4OutputError, match="unterminated"):
        encoder.finish()
