"""RED contracts for the canonical runtime Responses projection."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from mlx_batch_server.responses.controller import PreparedResponse
from mlx_batch_server.responses.runtime_projection import (
    RuntimeProjectionError,
    RuntimeResponseProjection,
    create_runtime_projection,
)
from mlx_batch_server.runtime.contracts import GenerationRequest, RuntimeKey
from mlx_batch_server.runtime.events import (
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    ReasoningCompleted,
    ReasoningDelta,
    SequencedTurnEvent,
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


def _prepared(
    response_id: str = "resp_projection",
    model: str = "buddy",
) -> PreparedResponse:
    return PreparedResponse(
        request=GenerationRequest(
            response_id=response_id,
            runtime=RuntimeKey(model_id=model),
            messages=({"role": "user", "content": "hello"},),
        ),
        materialized_messages=({"role": "user", "content": "hello"},),
    )


def _observe(
    projection: RuntimeResponseProjection,
    events: Iterable[TurnEvent],
) -> None:
    for sequence_number, event in enumerate(events):
        projection.observe(SequencedTurnEvent(sequence_number, event))


def _completed_text_events() -> tuple[TurnEvent, ...]:
    return (
        TurnStarted("resp_projection", "buddy", 123),
        OutputItemStarted("message", 0, "msg_1"),
        ContentPartStarted("output_text", 0, 0, "msg_1"),
        TextDelta("hel", "msg_1", 0, 0),
        TextDelta("lo", "msg_1", 0, 0),
        TextCompleted("hello", "msg_1", 0, 0),
        ContentPartCompleted("output_text", 0, 0, "msg_1", "hello"),
        OutputItemCompleted("message", 0, "msg_1", text="hello"),
        TurnCompleted("stop"),
    )


def test_mixed_reasoning_message_and_tool_projection_is_exact() -> None:
    projection = RuntimeResponseProjection(_prepared(), clock=lambda: 7.0)
    usage = UsageUpdate(
        11,
        13,
        24,
        cached_input_tokens=3,
        cache_write_input_tokens=2,
        reasoning_output_tokens=5,
    )
    events: tuple[TurnEvent, ...] = (
        TurnStarted("resp_projection", "buddy", 123),
        OutputItemStarted("reasoning", 0, "rs_1"),
        ContentPartStarted("reasoning_summary_text", 0, 0, "rs_1"),
        ReasoningDelta("chain ", "rs_1", 0, 0),
        ReasoningDelta("thought", "rs_1", 0, 0),
        ReasoningCompleted("chain thought", "rs_1", 0, 0),
        ContentPartCompleted("reasoning_summary_text", 0, 0, "rs_1", "chain thought"),
        OutputItemCompleted("reasoning", 0, "rs_1", text="chain thought"),
        OutputItemStarted("message", 1, "msg_1"),
        ContentPartStarted("output_text", 1, 0, "msg_1"),
        TextDelta("final ", "msg_1", 1, 0),
        TextDelta("answer", "msg_1", 1, 0),
        TextCompleted("final answer", "msg_1", 1, 0),
        ContentPartCompleted("output_text", 1, 0, "msg_1", "final answer"),
        OutputItemCompleted("message", 1, "msg_1", text="final answer"),
        OutputItemStarted("function_call", 2, "fc_1"),
        ToolDelta(2, "call_1", "fc_1", "lookup", '{"q":'),
        ToolDelta(2, "call_1", "fc_1", None, '"cats"}'),
        ToolCompleted(2, "call_1", "fc_1", "lookup", '{"q":"cats"}'),
        OutputItemCompleted(
            "function_call",
            2,
            "fc_1",
            call_id="call_1",
            name="lookup",
            arguments='{"q":"cats"}',
        ),
        usage,
        TurnCompleted("tool_calls", usage),
    )

    _observe(projection, events)

    assert projection.terminal_envelope() == {
        "id": "resp_projection",
        "object": "response",
        "status": "completed",
        "model": "buddy",
        "created_at": 123,
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "chain thought"}],
            },
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "final answer",
                        "annotations": [],
                    }
                ],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"cats"}',
            },
        ],
        "usage": {
            "input_tokens": 11,
            "input_tokens_details": {
                "cache_write_tokens": 2,
                "cached_tokens": 3,
            },
            "output_tokens": 13,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 24,
        },
        "error": None,
        "incomplete_details": None,
    }


def test_failure_and_cancellation_have_controller_terminal_statuses() -> None:
    failed = RuntimeResponseProjection(_prepared(), clock=lambda: 42.0)
    failed.observe(
        SequencedTurnEvent(
            0,
            TurnFailed("backend unavailable", "backend_error", 503),
        )
    )

    assert failed.terminal_envelope() == {
        "id": "resp_projection",
        "object": "response",
        "status": "failed",
        "model": "buddy",
        "created_at": 42,
        "output": [],
        "usage": None,
        "error": {"code": "backend_error", "message": "backend unavailable"},
        "incomplete_details": None,
    }

    cancelled = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
    _observe(
        cancelled,
        (
            TurnStarted("resp_projection", "buddy", 43),
            UsageUpdate(2, 1, 3),
            TurnCancelled("client_cancelled"),
        ),
    )
    terminal = cancelled.terminal_envelope()
    assert terminal["status"] == "cancelled"
    assert terminal["created_at"] == 43
    assert terminal["usage"] == {
        "input_tokens": 2,
        "input_tokens_details": {
            "cache_write_tokens": 0,
            "cached_tokens": 0,
        },
        "output_tokens": 1,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 3,
    }
    assert terminal["error"] is None
    assert terminal["incomplete_details"] is None


def test_length_finish_projects_incomplete_response_and_output_item() -> None:
    projection = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
    usage = UsageUpdate(4, 2, 6)
    _observe(
        projection,
        (
            TurnStarted("resp_projection", "buddy", 43),
            OutputItemStarted("message", 0, "msg_limited"),
            ContentPartStarted("output_text", 0, 0, "msg_limited"),
            TextDelta("cut off", "msg_limited", 0, 0),
            TextCompleted("cut off", "msg_limited", 0, 0),
            ContentPartCompleted("output_text", 0, 0, "msg_limited", "cut off"),
            OutputItemCompleted(
                "message",
                0,
                "msg_limited",
                text="cut off",
                status="incomplete",
            ),
            usage,
            TurnCompleted("length", usage),
        ),
    )

    terminal = projection.terminal_envelope()
    assert terminal["status"] == "incomplete"
    assert terminal["incomplete_details"] == {"reason": "max_output_tokens"}
    assert terminal["output"][0]["status"] == "incomplete"
    assert terminal["usage"]["output_tokens"] == 2


def test_terminal_envelope_is_deeply_immutable_and_factory_compatible() -> None:
    prepared = _prepared()
    projection = create_runtime_projection(prepared)
    assert isinstance(projection, RuntimeResponseProjection)
    _observe(projection, _completed_text_events())
    terminal = projection.terminal_envelope()

    with pytest.raises(TypeError, match="immutable"):
        terminal["status"] = "failed"  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        terminal["output"].append({})  # type: ignore[union-attr]
    with pytest.raises(TypeError, match="immutable"):
        terminal["output"][0]["content"][0]["text"] = "changed"

    assert projection.terminal_envelope()["output"][0]["content"][0]["text"] == "hello"


def test_sequence_and_response_identity_mismatches_fail_closed() -> None:
    projection = RuntimeResponseProjection(_prepared())
    projection.observe(
        SequencedTurnEvent(4, TurnStarted("resp_projection", "buddy", 10))
    )
    with pytest.raises(RuntimeProjectionError, match="monotonically"):
        projection.observe(SequencedTurnEvent(4, UsageUpdate(1, 1, 2)))

    wrong_response = RuntimeResponseProjection(_prepared())
    with pytest.raises(RuntimeProjectionError, match="response_id"):
        wrong_response.observe(
            SequencedTurnEvent(0, TurnStarted("resp_other", "buddy", 10))
        )

    wrong_model = RuntimeResponseProjection(_prepared())
    with pytest.raises(RuntimeProjectionError, match="model"):
        wrong_model.observe(
            SequencedTurnEvent(0, TurnStarted("resp_projection", "other", 10))
        )


def test_item_identity_and_done_payload_mismatches_fail_closed() -> None:
    projection = RuntimeResponseProjection(_prepared())
    _observe(
        projection,
        (
            TurnStarted("resp_projection", "buddy", 10),
            OutputItemStarted("message", 0, "msg_1"),
            ContentPartStarted("output_text", 0, 0, "msg_1"),
        ),
    )

    with pytest.raises(RuntimeProjectionError, match="item id"):
        projection.observe(SequencedTurnEvent(3, TextDelta("hello", "msg_other", 0, 0)))

    projection.observe(SequencedTurnEvent(3, TextDelta("hello", "msg_1", 0, 0)))
    with pytest.raises(RuntimeProjectionError, match="does not match"):
        projection.observe(
            SequencedTurnEvent(4, TextCompleted("goodbye", "msg_1", 0, 0))
        )
