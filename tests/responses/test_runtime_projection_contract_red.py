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
        OutputItemStarted("function_call", 2, "fc_1", "call_1", "lookup"),
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
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
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
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
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

    steered = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
    _observe(
        steered,
        (
            TurnStarted("resp_projection", "buddy", 44),
            TurnCancelled("steered"),
        ),
    )
    steered_terminal = steered.terminal_envelope()
    assert steered_terminal["status"] == "incomplete"
    assert steered_terminal["incomplete_details"] == {"reason": "steered"}


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


_PHYSICAL_MODEL = "grant-ai/Qwen3.8-Flash-Next"


def _aliased_prepared() -> PreparedResponse:
    """A `buddy` request resolved onto a physical runtime, as production does."""

    return PreparedResponse(
        request=GenerationRequest(
            response_id="resp_alias",
            runtime=RuntimeKey(model_id=_PHYSICAL_MODEL),
            messages=({"role": "user", "content": "hello"},),
            tools=(
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            ),
            sampling={"temperature": 0.2, "tool_choice": "auto"},
            metadata={
                "requested_model": "buddy",
                "resolved_model": _PHYSICAL_MODEL,
                "runtime_role": "main",
            },
        ),
        materialized_messages=({"role": "user", "content": "hello"},),
    )


def test_terminal_publishes_the_alias_while_the_runtime_validates_the_physical() -> (
    None
):
    """One public value, one internal value, and the runtime still checks its own."""

    projection = RuntimeResponseProjection(_aliased_prepared(), clock=lambda: 5.0)
    _observe(
        projection,
        (
            TurnStarted("resp_alias", _PHYSICAL_MODEL, 5),
            TurnCompleted("stop"),
        ),
    )
    terminal = projection.terminal_envelope()

    assert terminal["model"] == "buddy"
    assert _PHYSICAL_MODEL not in repr(terminal)
    assert terminal["tools"][0]["name"] == "lookup"
    assert terminal["temperature"] == 0.2

    physical_witness = RuntimeResponseProjection(_aliased_prepared())
    with pytest.raises(RuntimeProjectionError, match="model does not match"):
        physical_witness.observe(
            SequencedTurnEvent(0, TurnStarted("resp_alias", "buddy", 5))
        )


def test_internal_runtime_resolution_never_crosses_the_public_boundary() -> None:
    projection = RuntimeResponseProjection(_aliased_prepared(), clock=lambda: 5.0)
    _observe(
        projection,
        (TurnStarted("resp_alias", _PHYSICAL_MODEL, 5), TurnCompleted("stop")),
    )
    published = repr(projection.terminal_envelope())

    for leaked in ("resolved_model", "runtime_role", _PHYSICAL_MODEL):
        assert leaked not in published


def test_turn_started_alias_must_agree_with_the_prepared_request() -> None:
    projection = RuntimeResponseProjection(_aliased_prepared())
    with pytest.raises(RuntimeProjectionError, match="requested_model does not match"):
        projection.observe(
            SequencedTurnEvent(
                0,
                TurnStarted(
                    "resp_alias",
                    _PHYSICAL_MODEL,
                    5,
                    requested_model="someone_elses_alias",
                ),
            )
        )


def test_terminal_envelope_validates_against_the_installed_sdk() -> None:
    from openai.types.responses import Response

    projection = RuntimeResponseProjection(_aliased_prepared(), clock=lambda: 5.0)
    _observe(
        projection,
        (
            TurnStarted("resp_alias", _PHYSICAL_MODEL, 5),
            OutputItemStarted("function_call", 0, "fc_1", "call_1", "lookup"),
            ToolDelta(0, "call_1", "fc_1", "lookup", "{}"),
            ToolCompleted(0, "call_1", "fc_1", "lookup", "{}"),
            OutputItemCompleted(
                "function_call",
                0,
                "fc_1",
                call_id="call_1",
                name="lookup",
                arguments="{}",
            ),
            TurnCompleted("tool_calls"),
        ),
    )
    validated = Response.model_validate(dict(projection.terminal_envelope()))
    assert validated.model == "buddy"
    assert validated.output[0].call_id == "call_1"


def test_tool_arguments_cannot_introduce_or_change_admitted_identity() -> None:
    projection = RuntimeResponseProjection(_aliased_prepared())
    _observe(
        projection,
        (
            TurnStarted("resp_alias", _PHYSICAL_MODEL, 5),
            OutputItemStarted("function_call", 0, "fc_1", "call_1", "lookup"),
        ),
    )
    with pytest.raises(RuntimeProjectionError, match="call id changed"):
        projection.observe(
            SequencedTurnEvent(9, ToolDelta(0, "foreign", "fc_1", "lookup", "{}"))
        )
    with pytest.raises(RuntimeProjectionError, match="name changed"):
        projection.observe(
            SequencedTurnEvent(10, ToolDelta(0, "call_1", "fc_1", "other", "{}"))
        )


# --- W3-HR2-3: hosted lifecycle facts in the terminal projection --------------

from mlx_batch_server.runtime.events import (  # noqa: E402
    HostedCallCompleted,
    HostedCallProgress,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
)

_SEALED_ACTION = {
    "kind": "search",
    "query": "mlx batch server",
    "sources": ("https://a.example/one",),
}
_HOSTED_RESULT = {
    "kind": "search_results",
    "digest": "digest-1",
    "query": "mlx",
    "results": (
        {
            "url": "https://a.example/one",
            "title": "One",
            "snippet": "source passage",
        },
    ),
}


def _hosted_lifecycle_events(
    *,
    status: str = "completed",
    with_result: bool = True,
) -> tuple[TurnEvent, ...]:
    action = dict(_SEALED_ACTION)
    if status == "failed":
        action["sources"] = ()
    events: list[TurnEvent] = [
        TurnStarted("resp_projection", "buddy", 123),
        OutputItemStarted(
            "hosted_call",
            0,
            "ws_1",
            "call_ws",
            "web_search",
            {"query": "mlx"},
        ),
        HostedCallStarted(0, "ws_1", "call_ws", "web_search", {"query": "mlx"}),
        HostedCallProgress(0, "ws_1", "call_ws", "searching"),
    ]
    if with_result:
        events.append(
            HostedCallResult(0, "ws_1", "call_ws", "web_search", _HOSTED_RESULT)
        )
    events.extend(
        (
            HostedCallCompleted(
                0,
                "ws_1",
                "call_ws",
                "web_search",
                status,
                {"call_id": "call_ws"},
            ),
            OutputItemCompleted(
                "hosted_call",
                0,
                "ws_1",
                call_id="call_ws",
                name="web_search",
                status=status,
                action=action,
            ),
            TurnCompleted("stop"),
        )
    )
    return tuple(events)


def test_hosted_terminal_item_renders_exactly_from_the_sealed_action() -> None:
    from openai.types.responses import Response

    from mlx_batch_server.responses.transport import render_completed_item

    projection = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
    events = _hosted_lifecycle_events()
    _observe(projection, events)
    terminal = projection.terminal_envelope()

    completed = next(
        event for event in events if isinstance(event, OutputItemCompleted)
    )
    expected_item = {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "mlx batch server",
            "sources": [{"type": "url", "url": "https://a.example/one"}],
        },
    }
    assert list(terminal["output"]) == [expected_item]
    # The stream item and the terminal item come from one shared renderer.
    assert render_completed_item(completed) == expected_item
    Response.model_validate(dict(terminal))

    failed = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
    _observe(failed, _hosted_lifecycle_events(status="failed", with_result=False))
    failed_item = failed.terminal_envelope()["output"][0]
    assert failed_item["status"] == "failed"
    assert list(failed_item["action"]["sources"]) == []


def test_hosted_result_content_never_enters_the_terminal_envelope() -> None:
    projection = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
    _observe(projection, _hosted_lifecycle_events())
    published = repr(projection.terminal_envelope())
    assert "digest-1" not in published
    assert "title" not in published


def test_hosted_lifecycle_identity_and_order_violations_fail_closed() -> None:
    def start() -> RuntimeResponseProjection:
        projection = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
        _observe(
            projection,
            (
                TurnStarted("resp_projection", "buddy", 123),
                OutputItemStarted(
                    "hosted_call",
                    0,
                    "ws_1",
                    "call_ws",
                    "web_search",
                    {"query": "mlx"},
                ),
            ),
        )
        return projection

    foreign_call = start()
    with pytest.raises(RuntimeProjectionError, match="call id"):
        foreign_call.observe(
            SequencedTurnEvent(
                7, HostedCallStarted(0, "ws_1", "call_other", "web_search", {})
            )
        )

    wrong_tool = start()
    with pytest.raises(RuntimeProjectionError, match="tool name"):
        wrong_tool.observe(
            SequencedTurnEvent(
                7, HostedCallStarted(0, "ws_1", "call_ws", "web_fetch", {})
            )
        )

    no_receipt = start()
    with pytest.raises(RuntimeProjectionError, match="receipt event first"):
        no_receipt.observe(
            SequencedTurnEvent(
                7,
                OutputItemCompleted(
                    "hosted_call",
                    0,
                    "ws_1",
                    call_id="call_ws",
                    name="web_search",
                    status="completed",
                    action=_SEALED_ACTION,
                ),
            )
        )

    status_mismatch = start()
    status_mismatch.observe(
        SequencedTurnEvent(
            7,
            HostedCallCompleted(0, "ws_1", "call_ws", "web_search", "failed", {}),
        )
    )
    with pytest.raises(RuntimeProjectionError, match="status does not match"):
        status_mismatch.observe(
            SequencedTurnEvent(
                8,
                OutputItemCompleted(
                    "hosted_call",
                    0,
                    "ws_1",
                    call_id="call_ws",
                    name="web_search",
                    status="completed",
                    action=_SEALED_ACTION,
                ),
            )
        )

    duplicate_receipt = start()
    duplicate_receipt.observe(
        SequencedTurnEvent(
            7,
            HostedCallCompleted(0, "ws_1", "call_ws", "web_search", "completed", {}),
        )
    )
    with pytest.raises(RuntimeProjectionError, match="already recorded"):
        duplicate_receipt.observe(
            SequencedTurnEvent(
                8,
                HostedCallCompleted(
                    0, "ws_1", "call_ws", "web_search", "completed", {}
                ),
            )
        )


def test_interrupted_hosted_call_is_omitted_rather_than_fabricated() -> None:
    """A hosted call without its sealed completion has no proven action."""

    projection = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
    _observe(
        projection,
        (
            TurnStarted("resp_projection", "buddy", 123),
            OutputItemStarted(
                "hosted_call",
                0,
                "ws_1",
                "call_ws",
                "web_search",
                {"query": "mlx"},
            ),
            HostedCallStarted(0, "ws_1", "call_ws", "web_search", {"query": "mlx"}),
            TurnCancelled("client_cancelled"),
        ),
    )
    terminal = projection.terminal_envelope()
    assert terminal["status"] == "cancelled"
    assert list(terminal["output"]) == []


def _citation_projection() -> RuntimeResponseProjection:
    projection = RuntimeResponseProjection(_prepared(), clock=lambda: 1.0)
    _observe(
        projection,
        (
            TurnStarted("resp_projection", "buddy", 123),
            OutputItemStarted(
                "hosted_call",
                0,
                "ws_1",
                "call_ws",
                "web_search",
                {"query": "mlx"},
            ),
            HostedCallStarted(0, "ws_1", "call_ws", "web_search", {"query": "mlx"}),
            HostedCallProgress(0, "ws_1", "call_ws", "executing"),
            HostedCallResult(0, "ws_1", "call_ws", "web_search", _HOSTED_RESULT),
            HostedCallCompleted(
                0,
                "ws_1",
                "call_ws",
                "web_search",
                "completed",
                {"call_id": "call_ws"},
            ),
            OutputItemCompleted(
                "hosted_call",
                0,
                "ws_1",
                call_id="call_ws",
                name="web_search",
                action=_SEALED_ACTION,
            ),
            OutputItemStarted("message", 1, "msg_1"),
            ContentPartStarted("output_text", 1, 0, "msg_1"),
            TextDelta("source passage", "msg_1", 1, 0),
        ),
    )
    return projection


def _citation(**overrides: object) -> HostedCitation:
    fields: dict[str, object] = {
        "output_index": 1,
        "item_id": "msg_1",
        "content_index": 0,
        "source_call_id": "call_ws",
        "source_url": "https://a.example/one",
        "cited_text": "source passage",
        "source_start": 0,
        "source_end": 14,
        "output_start": 0,
        "output_end": 14,
    }
    fields.update(overrides)
    return HostedCitation(**fields)  # type: ignore[arg-type]


def test_hosted_citation_is_validated_and_carried_into_terminal_snapshot() -> None:
    from openai.types.responses import Response

    projection = _citation_projection()
    tail: tuple[TurnEvent, ...] = (
        _citation(),
        TextCompleted("source passage", "msg_1", 1, 0),
        ContentPartCompleted("output_text", 1, 0, "msg_1", "source passage"),
        OutputItemCompleted("message", 1, "msg_1", text="source passage"),
        TurnCompleted("stop"),
    )
    for sequence_number, event in enumerate(tail, start=10):
        projection.observe(SequencedTurnEvent(sequence_number, event))

    annotation = projection.terminal_envelope()["output"][1]["content"][0][
        "annotations"
    ][0]
    assert annotation == {
        "type": "url_citation",
        "url": "https://a.example/one",
        "title": "https://a.example/one",
        "start_index": 0,
        "end_index": 14,
    }
    Response.model_validate(dict(projection.terminal_envelope()))
    assert "source passage" not in repr(projection._hosted_source_lengths)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"source_call_id": "unknown"}, "unknown hosted result"),
        ({"source_url": "https://unknown.example"}, "not proven"),
        ({"source_end": 99}, "source range"),
        ({"output_end": 99}, "output range"),
        ({"cited_text": "wrong"}, "does not match"),
    ],
)
def test_hosted_citation_mutations_fail_closed(
    overrides: dict[str, object],
    match: str,
) -> None:
    projection = _citation_projection()
    with pytest.raises(RuntimeProjectionError, match=match):
        projection.observe(SequencedTurnEvent(20, _citation(**overrides)))


def test_overlapping_and_late_hosted_citations_fail_closed() -> None:
    overlapping = _citation_projection()
    overlapping.observe(SequencedTurnEvent(20, _citation()))
    with pytest.raises(RuntimeProjectionError, match="must not overlap"):
        overlapping.observe(
            SequencedTurnEvent(
                21,
                _citation(
                    cited_text="passage",
                    source_start=7,
                    source_end=14,
                    output_start=7,
                    output_end=14,
                ),
            )
        )

    late = _citation_projection()
    late.observe(SequencedTurnEvent(20, TextCompleted("source passage", "msg_1", 1, 0)))
    with pytest.raises(RuntimeProjectionError, match="cannot follow"):
        late.observe(SequencedTurnEvent(21, _citation()))
