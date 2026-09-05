"""RED contracts for SSE and multiplexed Responses WebSocket transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from mlx_batch_server.responses.projector import (
    encode_sse,
    encode_websocket,
    project_event,
)
from mlx_batch_server.responses.transport import (
    MultiplexedTransportSession,
    QueueCapacityError,
    ResponseCreateCommand,
    ResponseEventSource,
    ResponseInjectCommand,
    ResponseSteerCommand,
    StreamCapacityError,
    StreamId,
    TransportEnvelope,
    TransportErrorOutcome,
    TransportProtocolError,
    TransportSession,
    UnknownStreamError,
    parse_websocket_command,
)
from mlx_batch_server.responses.websocket import (
    ResponsesWebSocketSession,
    render_protocol_error,
)
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
    TurnStarted,
    UsageUpdate,
)


def _session(
    connection_id: str = "connection-1",
    *,
    max_active_responses: int = 16,
    max_named_streams: int = 32,
    max_queued_responses: int = 128,
    max_pending_events_per_stream: int = 32,
) -> TransportSession:
    return TransportSession(
        connection_id=connection_id,
        principal="founder",
        opened_at=1.0,
        max_active_responses=max_active_responses,
        max_named_streams=max_named_streams,
        max_queued_responses=max_queued_responses,
        max_pending_events_per_stream=max_pending_events_per_stream,
    )


async def _events(*events: TurnEvent) -> AsyncIterator[SequencedTurnEvent]:
    for sequence_number, event in enumerate(events):
        yield SequencedTurnEvent(sequence_number, event)


def _source(response_id: str, text: str = "ok") -> ResponseEventSource:
    item_id = f"msg_{response_id}"
    return ResponseEventSource(
        _events(
            TurnStarted(response_id=response_id, model="buddy", created_at=1),
            OutputItemStarted(kind="message", index=0, item_id=item_id),
            ContentPartStarted(
                kind="output_text",
                output_index=0,
                content_index=0,
                item_id=item_id,
            ),
            TextDelta(
                delta=text,
                item_id=item_id,
                output_index=0,
                content_index=0,
            ),
            TextCompleted(
                text=text,
                item_id=item_id,
                output_index=0,
                content_index=0,
            ),
            ContentPartCompleted(
                kind="output_text",
                output_index=0,
                content_index=0,
                item_id=item_id,
                text=text,
            ),
            OutputItemCompleted(kind="message", index=0, item_id=item_id, text=text),
            TurnCompleted("stop"),
        ),
        response_id=response_id,
    )


def test_command_parser_uses_flat_response_create_envelope() -> None:
    create = parse_websocket_command(
        {
            "type": "response.create",
            "stream_id": "buddy.round-1",
            "model": "buddy",
            "input": "hello",
            "previous_response_id": "resp_parent",
            "store": False,
        }
    )

    assert isinstance(create, ResponseCreateCommand)
    assert create.stream_id == StreamId("buddy.round-1")
    assert create.response == {
        "model": "buddy",
        "input": "hello",
        "previous_response_id": "resp_parent",
        "store": False,
    }
    assert "stream_id" not in create.response
    assert create.response["previous_response_id"] == "resp_parent"


@pytest.mark.parametrize("field", ["stream", "background"])
def test_response_create_rejects_transport_fields(field: str) -> None:
    with pytest.raises(TransportProtocolError) as raised:
        parse_websocket_command(
            {"type": "response.create", "model": "buddy", field: False}
        )

    assert raised.value.param == field


def test_default_lane_and_stream_id_validation_match_websocket_contract() -> None:
    default = parse_websocket_command(
        {"type": "response.create", "model": "buddy", "input": "hello"}
    )
    assert isinstance(default, ResponseCreateCommand)
    assert default.stream_id is None

    assert StreamId("A_z-9.ok").value == "A_z-9.ok"
    assert len(StreamId("a" * 256).value) == 256
    for invalid in ("", "a" * 257, "contains whitespace", "slash/not-allowed"):
        with pytest.raises(ValueError):
            StreamId(invalid)


def test_response_steer_parser_accepts_only_official_user_input_shape() -> None:
    steer = parse_websocket_command(
        {
            "type": "response.steer",
            "previous_response_id": "resp_active",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "look here",
                            "prompt_cache_breakpoint": {"type": "default"},
                        },
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AA==",
                            "detail": "original",
                        },
                        {
                            "type": "input_file",
                            "file_id": "file_123",
                            "detail": "high",
                        },
                    ],
                }
            ],
        }
    )

    assert isinstance(steer, ResponseSteerCommand)
    assert steer.previous_response_id == "resp_active"
    assert steer.input[0]["role"] == "user"


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        (
            {
                "type": "response.steer",
                "stream_id": "main",
                "previous_response_id": "resp_active",
                "input": "redirect",
            },
            "stream_id",
        ),
        (
            {
                "type": "response.steer",
                "previous_response_id": None,
                "input": "redirect",
            },
            "previous_response_id",
        ),
        (
            {
                "type": "response.steer",
                "previous_response_id": "resp_active",
                "input": [],
            },
            "input",
        ),
        (
            {
                "type": "response.steer",
                "previous_response_id": "resp_active",
                "input": [{"role": "assistant", "content": "no"}],
            },
            "input[0].role",
        ),
        (
            {
                "type": "response.steer",
                "previous_response_id": "resp_active",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "output_text", "text": "not input"}],
                    }
                ],
            },
            "input[0].content[0].type",
        ),
        (
            {
                "type": "response.steer",
                "previous_response_id": "resp_active",
                "input": [{"role": "user", "content": "ok", "id": "msg_1"}],
            },
            "input[0].id",
        ),
        (
            {
                "type": "response.steer",
                "previous_response_id": "resp_active",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_data": "data:application/pdf;base64,AAAA",
                                "detail": "extreme",
                            }
                        ],
                    }
                ],
            },
            "input[0].content[0].detail",
        ),
    ],
)
def test_response_steer_parser_rejects_non_contract_fields_and_items(
    payload: dict[str, object],
    param: str,
) -> None:
    with pytest.raises(TransportProtocolError) as raised:
        parse_websocket_command(payload)

    assert raised.value.param == param


def test_custom_cancel_and_ping_are_not_websocket_client_events() -> None:
    for payload in (
        {"type": "response.cancel", "stream_id": "main"},
        {"type": "ping", "nonce": "legacy"},
    ):
        with pytest.raises(TransportProtocolError) as raised:
            parse_websocket_command(payload)
        assert raised.value.param == "type"


def test_beta_response_inject_is_parsed_only_for_fail_closed_rejection() -> None:
    command = parse_websocket_command(
        {
            "type": "response.inject",
            "response_id": "resp_active",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done",
                }
            ],
        }
    )

    assert isinstance(command, ResponseInjectCommand)
    assert command.response_id == "resp_active"


def test_protocol_errors_use_official_error_shape_and_echo_known_stream() -> None:
    error = TransportProtocolError(
        "background is not available in WebSocket mode",
        stream_id=StreamId("main"),
        param="background",
    )

    assert render_protocol_error(error) == {
        "type": "error",
        "status": 400,
        "stream_id": "main",
        "error": {
            "type": "invalid_request_error",
            "code": "transport_protocol_error",
            "message": "background is not available in WebSocket mode",
            "param": "background",
        },
    }


@pytest.mark.asyncio
async def test_handle_payload_renders_named_create_errors_without_closing() -> None:
    websocket = ResponsesWebSocketSession(_session())

    rendered = await websocket.handle_payload(
        {
            "type": "response.create",
            "stream_id": "main",
            "model": "buddy",
            "background": False,
        }
    )

    assert rendered is not None
    assert rendered["type"] == "error"
    assert rendered["status"] == 400
    assert rendered["stream_id"] == "main"
    assert rendered["error"]["param"] == "background"
    assert not websocket.closed


@pytest.mark.asyncio
async def test_beta_response_inject_fails_closed_without_atomic_injection_seam() -> (
    None
):
    websocket = ResponsesWebSocketSession(_session())

    rendered = await websocket.handle_payload(
        {
            "type": "response.inject",
            "response_id": "resp_active",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "done",
                }
            ],
        }
    )

    assert rendered == {
        "type": "error",
        "status": 400,
        "error": {
            "type": "invalid_request_error",
            "code": "response_inject_not_supported",
            "message": (
                "Beta Multi-agent response.inject is unsupported without an "
                "atomic active-response injection seam"
            ),
            "param": "type",
        },
    }
    assert websocket.closed


@pytest.mark.asyncio
async def test_malformed_beta_response_inject_also_fails_closed() -> None:
    websocket = ResponsesWebSocketSession(_session())

    rendered = await websocket.handle_payload(
        {
            "type": "response.inject",
            "response_id": "resp_active",
            "input": [],
        }
    )

    assert rendered is not None
    assert rendered["type"] == "error"
    assert rendered["status"] == 400
    assert rendered["error"]["param"] == "input"
    assert websocket.closed


def test_named_events_echo_stream_id_and_default_events_omit_it() -> None:
    event = TextDelta(
        delta="beautiful output",
        item_id="msg_1",
        output_index=0,
        content_index=0,
    )
    named = project_event(TransportEnvelope(StreamId("buddy"), 7, event))
    default = project_event(TransportEnvelope(None, 7, event))

    assert named["stream_id"] == "buddy"
    assert "stream_id" not in default
    assert named["sequence_number"] == default["sequence_number"] == 7


def test_stream_id_is_websocket_only_while_event_payloads_otherwise_match() -> None:
    envelope = TransportEnvelope(
        stream_id=StreamId("buddy"),
        sequence_number=0,
        event=TurnStarted(response_id="resp_test", model="buddy", created_at=1),
    )
    projected = project_event(envelope)
    sse = encode_sse(envelope).decode()
    data_line = next(line for line in sse.splitlines() if line.startswith("data: "))

    sse_payload = json.loads(data_line.removeprefix("data: "))
    websocket_payload = json.loads(encode_websocket(envelope))
    assert "stream_id" not in sse_payload
    assert websocket_payload.pop("stream_id") == "buddy"
    assert websocket_payload == sse_payload
    assert projected["stream_id"] == "buddy"
    assert projected["type"] == "response.created"


def test_projector_uses_official_item_content_delta_and_done_types() -> None:
    stream_id = StreamId("reasoning")
    cases = (
        (
            OutputItemStarted("message", 0, "msg_1"),
            "response.output_item.added",
        ),
        (
            ContentPartStarted("output_text", 0, 0, "msg_1"),
            "response.content_part.added",
        ),
        (
            TextDelta("hi", "msg_1", 0, 0),
            "response.output_text.delta",
        ),
        (
            TextCompleted("hi", "msg_1", 0, 0),
            "response.output_text.done",
        ),
        (
            ContentPartCompleted("output_text", 0, 0, "msg_1", "hi"),
            "response.content_part.done",
        ),
        (
            OutputItemCompleted("message", 0, "msg_1", text="hi"),
            "response.output_item.done",
        ),
        (
            ContentPartStarted("reasoning_summary_text", 1, 0, "rs_1"),
            "response.reasoning_summary_part.added",
        ),
        (
            ReasoningDelta("why", "rs_1", 1, 0),
            "response.reasoning_summary_text.delta",
        ),
        (
            ReasoningCompleted("why", "rs_1", 1, 0),
            "response.reasoning_summary_text.done",
        ),
        (
            ContentPartCompleted("reasoning_summary_text", 1, 0, "rs_1", "why"),
            "response.reasoning_summary_part.done",
        ),
        (
            ToolDelta(2, "call_1", "fc_1", "lookup", "{"),
            "response.function_call_arguments.delta",
        ),
        (
            ToolCompleted(2, "call_1", "fc_1", "lookup", "{}"),
            "response.function_call_arguments.done",
        ),
        (
            OutputItemCompleted(
                "function_call",
                2,
                "fc_1",
                call_id="call_1",
                name="lookup",
                arguments="{}",
            ),
            "response.output_item.done",
        ),
    )

    projected_by_type: dict[str, list[dict[str, Any]]] = {}
    for sequence_number, (event, expected_type) in enumerate(cases):
        projected = project_event(TransportEnvelope(stream_id, sequence_number, event))
        assert projected["type"] == expected_type
        assert projected["stream_id"] == "reasoning"
        projected_by_type.setdefault(expected_type, []).append(projected)

    content_done = projected_by_type["response.content_part.done"][0]
    assert content_done["part"] == {
        "type": "output_text",
        "text": "hi",
        "annotations": [],
        "logprobs": [],
    }
    reasoning_done = projected_by_type["response.reasoning_summary_part.done"][0]
    assert reasoning_done["part"] == {"type": "summary_text", "text": "why"}
    message_done, function_done = projected_by_type["response.output_item.done"]
    assert message_done["item"]["content"][0]["text"] == "hi"
    assert function_done["item"] == {
        "id": "fc_1",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{}",
    }
    assert projected_by_type["response.output_text.delta"][0]["logprobs"] == []
    assert projected_by_type["response.output_text.done"][0]["logprobs"] == []

    cancelled = project_event(
        TransportEnvelope(stream_id, len(cases), TurnCancelled("user_stopped"))
    )
    assert cancelled["type"] == "response.incomplete"
    assert cancelled["response"]["status"] == "incomplete"


def test_length_finish_projects_official_incomplete_terminal() -> None:
    stream_id = StreamId("limited")
    output_done = project_event(
        TransportEnvelope(
            stream_id,
            7,
            OutputItemCompleted(
                "message",
                0,
                "msg_limited",
                text="cut off",
                status="incomplete",
            ),
        )
    )
    terminal = project_event(
        TransportEnvelope(
            stream_id,
            8,
            TurnCompleted("length", UsageUpdate(4, 2, 6)),
        )
    )

    assert output_done["item"]["status"] == "incomplete"
    assert terminal == {
        "sequence_number": 8,
        "stream_id": "limited",
        "type": "response.incomplete",
        "response": {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {
                    "cache_write_tokens": 0,
                    "cached_tokens": 0,
                },
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 6,
            },
        },
    }


@pytest.mark.asyncio
async def test_same_stream_id_queues_fifo_without_overlap() -> None:
    releases = {"first": asyncio.Event(), "second": asyncio.Event()}
    started: list[str] = []

    async def start(command: ResponseCreateCommand) -> ResponseEventSource:
        name = str(command.response["input"])
        started.append(name)

        async def held() -> AsyncIterator[SequencedTurnEvent]:
            yield SequencedTurnEvent(
                0,
                TurnStarted(response_id=f"resp_{name}", model="buddy", created_at=1),
            )
            await releases[name].wait()
            yield SequencedTurnEvent(1, TurnCompleted("stop"))

        return ResponseEventSource(held())

    websocket = ResponsesWebSocketSession(_session(), start)
    await websocket.create(
        ResponseCreateCommand(
            stream_id=StreamId("main"),
            response={"model": "buddy", "input": "first"},
        )
    )
    await websocket.create(
        ResponseCreateCommand(
            stream_id=StreamId("main"),
            response={"model": "buddy", "input": "second"},
        )
    )
    await asyncio.sleep(0)

    assert started == ["first"]
    assert websocket.session.max_active_responses == 16
    releases["first"].set()
    first_events = [await websocket.receive_payload() for _ in range(2)]
    assert [event["type"] for event in first_events] == [
        "response.created",
        "response.completed",
    ]
    await asyncio.sleep(0)
    assert started == ["first", "second"]
    releases["second"].set()
    second_events = [await websocket.receive_payload() for _ in range(2)]
    assert [event["sequence_number"] for event in second_events] == [0, 1]


@pytest.mark.asyncio
async def test_different_streams_start_concurrently_and_events_may_interleave() -> None:
    release = asyncio.Event()
    started: list[str] = []

    def source(name: str) -> ResponseEventSource:
        async def held() -> AsyncIterator[SequencedTurnEvent]:
            started.append(name)
            yield SequencedTurnEvent(
                0,
                TurnStarted(response_id=f"resp_{name}", model="buddy", created_at=1),
            )
            await release.wait()
            yield SequencedTurnEvent(1, TurnCompleted("stop"))

        return ResponseEventSource(held())

    core = MultiplexedTransportSession(_session())
    await core.open(StreamId("alpha"), lambda: source("alpha"))
    await core.open(StreamId("beta"), lambda: source("beta"))
    await asyncio.sleep(0)

    assert set(started) == {"alpha", "beta"}
    first = await core.receive()
    second = await core.receive()
    assert first.stream_id != second.stream_id
    release.set()
    terminals = [await core.receive(), await core.receive()]
    assert all(isinstance(item.event, TurnCompleted) for item in terminals)


@pytest.mark.asyncio
async def test_connection_queues_beyond_sixteen_active_responses() -> None:
    releases = [asyncio.Event() for _ in range(17)]
    started: list[int] = []
    core = MultiplexedTransportSession(_session())

    def source(index: int) -> ResponseEventSource:
        async def held() -> AsyncIterator[SequencedTurnEvent]:
            started.append(index)
            yield SequencedTurnEvent(
                0,
                TurnStarted(response_id=f"resp_{index}", model="buddy", created_at=1),
            )
            await releases[index].wait()
            yield SequencedTurnEvent(1, TurnCompleted("stop"))

        return ResponseEventSource(held())

    for index in range(17):
        await core.open(StreamId(f"lane-{index}"), lambda index=index: source(index))
    await asyncio.sleep(0)

    assert len(started) == 16
    assert core.active_response_count == 16
    assert core.queued_response_count == 1

    releases[0].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(started) == 17
    while True:
        event = await core.receive()
        if event.stream_id == StreamId("lane-0") and isinstance(
            event.event, TurnCompleted
        ):
            break
    await core.close()


@pytest.mark.asyncio
async def test_named_stream_limit_is_lifetime_scoped_and_excludes_default_lane() -> (
    None
):
    core = MultiplexedTransportSession(_session())
    await core.open(None, lambda: _source("resp_default"))
    for index in range(32):
        await core.open(
            StreamId(f"lane-{index}"),
            lambda index=index: _source(f"resp_{index}"),
        )

    assert len(core.named_stream_ids) == 32
    with pytest.raises(StreamCapacityError) as raised:
        await core.open(StreamId("lane-33"), lambda: _source("resp_rejected"))
    assert raised.value.code == "websocket_stream_limit_reached"
    assert raised.value.param == "stream_id"
    await core.close()


@pytest.mark.asyncio
async def test_response_queue_is_bounded() -> None:
    release = asyncio.Event()

    async def held() -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(response_id="resp_active", model="buddy", created_at=1),
        )
        await release.wait()
        yield SequencedTurnEvent(1, TurnCompleted("stop"))

    core = MultiplexedTransportSession(
        _session(max_active_responses=1, max_queued_responses=1)
    )
    await core.open(StreamId("active"), lambda: ResponseEventSource(held()))
    await core.open(StreamId("queued"), lambda: _source("resp_queued"))
    with pytest.raises(QueueCapacityError):
        await core.open(StreamId("overflow"), lambda: _source("resp_overflow"))
    release.set()
    await core.close()


@pytest.mark.asyncio
async def test_full_queue_does_not_block_an_idle_lane_with_global_capacity() -> None:
    release = asyncio.Event()
    started: list[str] = []

    def held(name: str) -> ResponseEventSource:
        async def events() -> AsyncIterator[SequencedTurnEvent]:
            started.append(name)
            yield SequencedTurnEvent(
                0,
                TurnStarted(response_id=f"resp_{name}", model="buddy", created_at=1),
            )
            await release.wait()
            yield SequencedTurnEvent(1, TurnCompleted("stop"))

        return ResponseEventSource(events())

    core = MultiplexedTransportSession(
        _session(max_active_responses=2, max_queued_responses=1)
    )
    await core.open(StreamId("busy"), lambda: held("busy"))
    await core.open(StreamId("busy"), lambda: held("queued"))
    assert core.queued_response_count == 1

    await core.open(StreamId("idle"), lambda: held("idle"))
    await asyncio.sleep(0)

    assert set(started) == {"busy", "idle"}
    assert core.queued_response_count == 1
    release.set()
    await core.close()


@pytest.mark.asyncio
async def test_response_sources_are_created_only_when_their_lane_starts() -> None:
    release = asyncio.Event()
    factories_called: list[str] = []

    def factory(name: str) -> ResponseEventSource:
        factories_called.append(name)

        async def events() -> AsyncIterator[SequencedTurnEvent]:
            yield SequencedTurnEvent(
                0,
                TurnStarted(response_id=f"resp_{name}", model="buddy", created_at=1),
            )
            await release.wait()
            yield SequencedTurnEvent(1, TurnCompleted("stop"))

        return ResponseEventSource(events())

    core = MultiplexedTransportSession(_session(max_active_responses=1))
    await core.open(StreamId("first"), lambda: factory("first"))
    await core.open(StreamId("second"), lambda: factory("second"))
    await asyncio.sleep(0)

    assert factories_called == ["first"]
    release.set()
    await core.close()


@pytest.mark.asyncio
async def test_transport_rejects_eager_response_event_sources() -> None:
    core = MultiplexedTransportSession(_session())

    with pytest.raises(TypeError, match="lazy factory"):
        await core.open(  # type: ignore[arg-type]
            StreamId("main"), _source("resp_eager")
        )


@pytest.mark.asyncio
async def test_cancel_and_close_never_cross_connection_or_lane() -> None:
    releases = {"alpha": asyncio.Event(), "beta": asyncio.Event()}
    cancelled: list[tuple[str, str]] = []

    def held(name: str) -> ResponseEventSource:
        async def events() -> AsyncIterator[SequencedTurnEvent]:
            yield SequencedTurnEvent(
                0,
                TurnStarted(response_id=f"resp_{name}", model="buddy", created_at=1),
            )
            await releases[name].wait()
            yield SequencedTurnEvent(1, TurnCompleted("stop"))

        return ResponseEventSource(
            events(),
            cancel=lambda reason: cancelled.append((name, reason)),
        )

    first = MultiplexedTransportSession(_session("first"))
    second = MultiplexedTransportSession(_session("second"))
    stream_id = StreamId("shared-name")
    await first.open(stream_id, lambda: held("alpha"))
    await second.open(stream_id, lambda: held("beta"))
    await asyncio.sleep(0)

    with pytest.raises(UnknownStreamError):
        await first.cancel(StreamId("foreign"), "wrong_lane")
    await first.cancel(stream_id, "user_stopped")
    assert cancelled == [("alpha", "user_stopped")]
    assert second.active_response_count == 1

    await first.close()
    assert cancelled == [("alpha", "user_stopped")]
    await second.close()
    assert cancelled == [
        ("alpha", "user_stopped"),
        ("beta", "transport_disconnected"),
    ]


@pytest.mark.asyncio
async def test_cancel_preserves_accepted_history_and_contiguous_sequence() -> None:
    source_waiting = asyncio.Event()
    cancel_requested = asyncio.Event()
    cancelled: list[str] = []

    async def events() -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(response_id="resp_main", model="buddy", created_at=1),
        )
        yield SequencedTurnEvent(1, TextDelta("kept", "msg_1", 0, 0))
        source_waiting.set()
        await cancel_requested.wait()
        yield SequencedTurnEvent(2, TurnCancelled("user_stopped"))

    def cancel(reason: str) -> None:
        cancelled.append(reason)
        cancel_requested.set()

    core = MultiplexedTransportSession(_session())
    await core.open(
        StreamId("main"),
        lambda: ResponseEventSource(events(), cancel=cancel),
    )
    await source_waiting.wait()

    await core.cancel(StreamId("main"), "user_stopped")
    observed = [await core.receive() for _ in range(3)]

    assert [item.sequence_number for item in observed] == [0, 1, 2]
    assert isinstance(observed[0].event, TurnStarted)
    assert isinstance(observed[1].event, TextDelta)
    assert isinstance(observed[2].event, TurnCancelled)
    assert cancelled == ["user_stopped"]


@pytest.mark.asyncio
async def test_cancel_waits_for_startup_then_reads_runtime_terminal() -> None:
    factory_started = asyncio.Event()
    allow_factory = asyncio.Event()
    cancel_requested = asyncio.Event()
    cancelled: list[str] = []

    async def factory() -> ResponseEventSource:
        factory_started.set()
        await allow_factory.wait()

        async def events() -> AsyncIterator[SequencedTurnEvent]:
            await cancel_requested.wait()
            yield SequencedTurnEvent(
                0,
                TurnCancelled("startup_cancelled"),
            )

        def cancel(reason: str) -> None:
            cancelled.append(reason)
            cancel_requested.set()

        return ResponseEventSource(events(), cancel=cancel)

    core = MultiplexedTransportSession(_session())
    await core.open(StreamId("main"), factory)
    await factory_started.wait()
    cancellation = asyncio.create_task(
        core.cancel(StreamId("main"), "startup_cancelled")
    )
    await asyncio.sleep(0)

    assert not cancellation.done()
    assert cancelled == []
    allow_factory.set()
    await cancellation
    terminal = await core.receive()

    assert cancelled == ["startup_cancelled"]
    assert terminal.sequence_number == 0
    assert isinstance(terminal.event, TurnCancelled)


@pytest.mark.asyncio
async def test_disconnect_detaches_sources_that_opt_out_of_cancellation() -> None:
    release = asyncio.Event()
    source_waiting = asyncio.Event()
    task_cancelled = asyncio.Event()
    cancel_callbacks: list[str] = []

    async def events() -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(response_id="resp_detached", model="buddy", created_at=1),
        )
        source_waiting.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            task_cancelled.set()
            raise
        yield SequencedTurnEvent(1, TurnCompleted("stop"))

    core = MultiplexedTransportSession(_session())
    await core.open(
        StreamId("detached"),
        lambda: ResponseEventSource(
            events(),
            cancel=cancel_callbacks.append,
            cancel_on_disconnect=False,
        ),
    )
    await source_waiting.wait()

    await core.close()
    assert cancel_callbacks == []
    assert not task_cancelled.is_set()

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not task_cancelled.is_set()


@pytest.mark.asyncio
async def test_backpressure_fails_only_the_slow_lane() -> None:
    cancelled: list[str] = []
    terminal_response = asyncio.get_running_loop().create_future()

    slow = ResponseEventSource(
        _events(
            TurnStarted(response_id="resp_slow", model="buddy", created_at=1),
            TextDelta("one", "msg_slow", 0, 0),
            TextDelta("two", "msg_slow", 0, 0),
            TurnCompleted("stop"),
        ),
        cancel=cancelled.append,
        terminal_response=terminal_response,
    )
    core = MultiplexedTransportSession(_session(max_pending_events_per_stream=2))
    await core.open(StreamId("slow"), lambda: slow)
    await core.open(
        StreamId("healthy"),
        lambda: ResponseEventSource(
            _events(
                TurnStarted(response_id="resp_ok", model="buddy", created_at=1),
                TurnCompleted("stop"),
            )
        ),
    )
    await asyncio.sleep(0)

    observed = [await core.receive() for _ in range(5)]
    slow_events = [item for item in observed if item.stream_id == StreamId("slow")]
    healthy_events = [
        item for item in observed if item.stream_id == StreamId("healthy")
    ]
    assert [
        item.sequence_number
        for item in slow_events
        if isinstance(item, TransportEnvelope)
    ] == [0, 1]
    assert isinstance(slow_events[0].event, TurnStarted)
    assert isinstance(slow_events[1].event, TextDelta)
    slow_fault = slow_events[-1]
    assert isinstance(slow_fault, TransportErrorOutcome)
    assert slow_fault.error.code == "transport_backpressure"
    assert slow_fault.error.status_code == 429
    assert slow_fault.terminal_response is terminal_response
    terminal = {
        "id": "resp_slow",
        "object": "response",
        "status": "failed",
        "output": [],
    }
    terminal_response.set_result(terminal)
    assert await slow_fault.terminal_response == terminal
    assert isinstance(healthy_events[-1].event, TurnCompleted)
    assert cancelled == ["transport_backpressure"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("items", "accepted_types"),
    [
        (
            (
                SequencedTurnEvent(
                    0,
                    TurnStarted(
                        response_id="resp_duplicate", model="buddy", created_at=1
                    ),
                ),
                SequencedTurnEvent(0, TextDelta("duplicate", "msg_1", 0, 0)),
            ),
            (TurnStarted,),
        ),
        (
            (
                SequencedTurnEvent(
                    0,
                    TurnStarted(
                        response_id="resp_decreasing", model="buddy", created_at=1
                    ),
                ),
                SequencedTurnEvent(1, TextDelta("kept", "msg_1", 0, 0)),
                SequencedTurnEvent(0, TextDelta("decreasing", "msg_1", 0, 0)),
            ),
            (TurnStarted, TextDelta),
        ),
        (
            (
                SequencedTurnEvent(
                    0,
                    TurnStarted(response_id="resp_gap", model="buddy", created_at=1),
                ),
                SequencedTurnEvent(2, TextDelta("gap", "msg_1", 0, 0)),
            ),
            (TurnStarted,),
        ),
        (
            (
                TurnStarted(response_id="resp_mixed_raw", model="buddy", created_at=1),
                SequencedTurnEvent(1, TurnCompleted("stop")),
            ),
            (TurnStarted,),
        ),
        (
            (
                SequencedTurnEvent(
                    0,
                    TurnStarted(
                        response_id="resp_mixed_sequenced",
                        model="buddy",
                        created_at=1,
                    ),
                ),
                TurnCompleted("stop"),
            ),
            (TurnStarted,),
        ),
    ],
)
async def test_source_sequence_violations_emit_one_transport_error_outcome(
    items: tuple[SequencedTurnEvent | TurnEvent, ...],
    accepted_types: tuple[type[TurnEvent], ...],
) -> None:
    cancelled: list[str] = []

    async def events() -> AsyncIterator[SequencedTurnEvent | TurnEvent]:
        for item in items:
            yield item

    core = MultiplexedTransportSession(_session())
    await core.open(
        StreamId("main"),
        lambda: ResponseEventSource(events(), cancel=cancelled.append),
    )
    observed = [await core.receive() for _ in range(len(accepted_types) + 1)]

    accepted = observed[:-1]
    assert all(isinstance(item, TransportEnvelope) for item in accepted)
    assert [type(item.event) for item in accepted] == list(accepted_types)
    assert [item.sequence_number for item in accepted] == list(range(len(accepted)))
    fault = observed[-1]
    assert isinstance(fault, TransportErrorOutcome)
    assert fault.error.code == "transport_sequence_error"
    assert not hasattr(fault, "sequence_number")
    assert cancelled == ["transport_sequence_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_kind", "expected_code"),
    [
        ("raises", "transport_source_failed"),
        ("ends", "missing_terminal_event"),
    ],
)
async def test_source_faults_request_cancel_without_authoring_response_terminal(
    source_kind: str,
    expected_code: str,
) -> None:
    cancelled: list[str] = []

    async def broken() -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(
                response_id=f"resp_{source_kind}",
                model="buddy",
                created_at=1,
            ),
        )
        if source_kind == "raises":
            raise RuntimeError("private source detail")

    core = MultiplexedTransportSession(_session())
    await core.open(
        StreamId("broken"),
        lambda: ResponseEventSource(
            broken(),
            cancel=cancelled.append,
        ),
    )

    accepted = await core.receive()
    fault = await core.receive()

    assert isinstance(accepted, TransportEnvelope)
    assert isinstance(accepted.event, TurnStarted)
    assert isinstance(fault, TransportErrorOutcome)
    assert fault.stream_id == StreamId("broken")
    assert fault.error.code == expected_code
    assert fault.error.error_type == "server_error"
    assert "private source detail" not in str(fault.error)
    assert cancelled == [expected_code]


@pytest.mark.asyncio
async def test_websocket_renders_lane_fault_as_official_error_not_response_event() -> (
    None
):
    async def invalid_sequence() -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(response_id="resp_main", model="buddy", created_at=1),
        )
        yield SequencedTurnEvent(2, TurnCompleted("stop"))

    websocket = ResponsesWebSocketSession(
        _session(),
        lambda command: ResponseEventSource(invalid_sequence()),
    )
    await websocket.create(
        ResponseCreateCommand(
            response={"model": "buddy", "input": "hello"},
            stream_id=StreamId("main"),
        )
    )

    created = await websocket.receive_payload()
    rendered = await websocket.receive_payload()

    assert created["type"] == "response.created"
    assert rendered == {
        "type": "error",
        "status": 500,
        "stream_id": "main",
        "error": {
            "type": "server_error",
            "code": "transport_sequence_error",
            "message": ("response event source violated contiguous sequence semantics"),
            "param": None,
        },
    }
    assert "sequence_number" not in rendered
    assert "response" not in rendered


@pytest.mark.asyncio
async def test_runtime_terminal_keeps_full_terminal_response_channel() -> None:
    terminal_response = asyncio.get_running_loop().create_future()
    full_terminal = {
        "id": "resp_full",
        "object": "response",
        "created_at": 1,
        "model": "buddy",
        "status": "completed",
        "output": [
            {
                "id": "msg_full",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "beautiful output",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 2,
            "output_tokens": 2,
            "total_tokens": 4,
        },
    }
    terminal_response.set_result(full_terminal)

    websocket = ResponsesWebSocketSession(
        _session(),
        lambda command: ResponseEventSource(
            _events(
                TurnStarted(
                    response_id="resp_full",
                    model="buddy",
                    created_at=1,
                ),
                TurnCompleted("stop"),
            ),
            terminal_response=terminal_response,
        ),
    )
    await websocket.create(
        ResponseCreateCommand(
            response={"model": "buddy", "input": "hello"},
            stream_id=StreamId("main"),
        )
    )

    await websocket.receive_payload()
    completed = await websocket.receive_payload()

    assert completed == {
        "type": "response.completed",
        "sequence_number": 1,
        "stream_id": "main",
        "response": full_terminal,
    }


def test_usage_projects_through_standard_in_progress_event() -> None:
    projected = project_event(
        TransportEnvelope(StreamId("main"), 3, UsageUpdate(2, 3, 5))
    )
    assert projected == {
        "sequence_number": 3,
        "stream_id": "main",
        "type": "response.in_progress",
        "response": {
            "status": "in_progress",
            "usage": {
                "input_tokens": 2,
                "input_tokens_details": {
                    "cache_write_tokens": 0,
                    "cached_tokens": 0,
                },
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5,
            },
        },
    }


@pytest.mark.asyncio
async def test_response_steer_dispatches_only_through_explicit_runtime_seam() -> None:
    observed: list[ResponseSteerCommand] = []
    websocket = ResponsesWebSocketSession(
        _session(),
        steer_response=observed.append,
    )

    rendered = await websocket.handle_payload(
        {
            "type": "response.steer",
            "previous_response_id": "resp_active",
            "input": "change direction",
        }
    )

    assert rendered is None
    assert observed == [
        ResponseSteerCommand(
            previous_response_id="resp_active",
            input="change direction",
        )
    ]


@pytest.mark.asyncio
async def test_response_steer_unknown_response_returns_uncommitted_input() -> None:
    websocket = ResponsesWebSocketSession(_session())

    rendered = await websocket.handle_payload(
        {
            "type": "response.steer",
            "previous_response_id": "resp_active",
            "input": "change direction",
        }
    )

    assert rendered is not None
    assert rendered == {
        "type": "response.steer.failed",
        "sequence_number": 0,
        "steer": {"previous_response_id": "resp_active"},
        "input": "change direction",
        "error": {
            "code": "response_not_found",
            "message": "the target response is not active on this connection",
        },
    }
    assert not websocket.closed


@pytest.mark.asyncio
async def test_response_steer_interrupts_parent_and_commits_inherited_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()
    cancel_reasons: list[str] = []
    commands: list[ResponseCreateCommand] = []

    async def parent_events() -> AsyncIterator[SequencedTurnEvent]:
        yield SequencedTurnEvent(
            0,
            TurnStarted(response_id="resp_parent", model="buddy", created_at=1),
        )
        await cancelled.wait()
        yield SequencedTurnEvent(1, TurnCancelled("steered"))

    def cancel_parent(reason: str) -> None:
        cancel_reasons.append(reason)
        cancelled.set()

    def start(command: ResponseCreateCommand) -> ResponseEventSource:
        commands.append(command)
        if len(commands) == 1:
            return ResponseEventSource(
                parent_events(),
                cancel=cancel_parent,
                response_id="resp_parent",
            )
        return _source("resp_successor", "new direction")

    def fixed_uuid() -> object:
        return type("_UUID", (), {"hex": "fixed"})()

    monkeypatch.setattr(
        "mlx_batch_server.responses.websocket.uuid.uuid4",
        fixed_uuid,
    )
    websocket = ResponsesWebSocketSession(_session(), start_response=start)
    await websocket.create(
        ResponseCreateCommand(
            response={
                "model": "buddy",
                "input": "original",
                "instructions": "Buddy policy",
                "temperature": 0.25,
            },
            stream_id=StreamId("main"),
        )
    )

    created = await websocket.receive_payload()
    assert created["type"] == "response.created"

    rendered = await websocket.handle_payload(
        {
            "type": "response.steer",
            "previous_response_id": "resp_parent",
            "input": "change direction",
        }
    )
    assert rendered is None

    accepted = await websocket.receive_payload()
    incomplete = await websocket.receive_payload()
    successor_created = await websocket.receive_payload()

    assert accepted == {
        "type": "response.steer.accepted",
        "sequence_number": 1,
        "stream_id": "main",
        "steer": {
            "id": "steer_fixed",
            "previous_response_id": "resp_parent",
        },
    }
    assert incomplete["type"] == "response.incomplete"
    assert incomplete["sequence_number"] == 2
    assert incomplete["response"]["status"] == "incomplete"
    assert incomplete["response"]["incomplete_details"] == {"reason": "steered"}
    assert successor_created["type"] == "response.created"
    assert successor_created["sequence_number"] == 0
    assert successor_created["response"]["id"] == "resp_successor"
    assert cancel_reasons == ["steered"]
    assert commands[1] == ResponseCreateCommand(
        response={
            "model": "buddy",
            "input": "change direction",
            "instructions": "Buddy policy",
            "temperature": 0.25,
            "previous_response_id": "resp_parent",
        },
        stream_id=StreamId("main"),
    )


_PHYSICAL_MODEL = "grant-ai/Qwen3.8-Flash-Next"
_PUBLIC_ALIAS = "buddy"
_REQUEST_SETTINGS = {
    "tools": [
        {
            "type": "function",
            "name": "lookup",
            "description": "look one fact up",
            "parameters": {"type": "object", "properties": {}},
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "temperature": 0.2,
    "top_p": 0.9,
    "max_output_tokens": 256,
    "instructions": "answer briefly",
}


def _mixed_lifecycle_events() -> tuple[TurnEvent, ...]:
    """Reasoning, visible text and one tool call in a single alias-resolved turn."""

    usage = UsageUpdate(11, 13, 24, cached_input_tokens=3, reasoning_output_tokens=5)
    return (
        TurnStarted(
            "resp_mixed",
            _PHYSICAL_MODEL,
            123,
            requested_model=_PUBLIC_ALIAS,
            request_settings=_REQUEST_SETTINGS,
        ),
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


def _projected_stream(
    events: tuple[TurnEvent, ...] = (),
) -> tuple[dict[str, Any], ...]:
    """Project one whole turn exactly as the transport lane publishes it."""

    from mlx_batch_server.responses.transport import ResponseSnapshotBuilder

    builder = ResponseSnapshotBuilder()
    projected: list[dict[str, Any]] = []
    for sequence_number, event in enumerate(events or _mixed_lifecycle_events()):
        builder.observe(event)
        projected.append(
            project_event(
                TransportEnvelope(
                    StreamId("main"),
                    sequence_number,
                    event,
                    snapshot=builder.snapshot(event),
                )
            )
        )
    return tuple(projected)


_LIFECYCLE_EVENT_TYPES = frozenset(
    (
        "response.created",
        "response.in_progress",
        "response.completed",
        "response.incomplete",
        "response.failed",
    )
)


def test_every_lifecycle_snapshot_validates_against_the_installed_sdk() -> None:
    """The generated official types are the schema oracle for every snapshot."""

    from openai.types.responses import Response

    projected = _projected_stream()
    lifecycle = tuple(
        item for item in projected if item["type"] in _LIFECYCLE_EVENT_TYPES
    )
    assert {item["type"] for item in lifecycle} >= {
        "response.created",
        "response.in_progress",
        "response.completed",
    }
    for item in lifecycle:
        snapshot = item["response"]
        for required in (
            "id",
            "object",
            "created_at",
            "model",
            "status",
            "output",
            "parallel_tool_calls",
            "tool_choice",
            "tools",
        ):
            assert required in snapshot, (item["type"], required)
        Response.model_validate(snapshot)


def test_removing_a_required_request_setting_makes_the_snapshot_red() -> None:
    """The oracle must actually reject a partial snapshot, or it proves nothing."""

    from openai.types.responses import Response
    from pydantic import ValidationError

    created = next(
        item for item in _projected_stream() if item["type"] == "response.created"
    )
    for field in ("parallel_tool_calls", "tool_choice", "tools", "model"):
        partial = {
            key: value for key, value in created["response"].items() if key != field
        }
        with pytest.raises(ValidationError):
            Response.model_validate(partial)


def test_added_output_items_validate_and_carry_no_completed_content() -> None:
    """An added item is complete in shape and empty in content."""

    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseReasoningItem,
    )
    from pydantic import ValidationError

    added = {
        item["item"]["type"]: item["item"]
        for item in _projected_stream()
        if item["type"] == "response.output_item.added"
    }
    assert set(added) == {"reasoning", "message", "function_call"}

    ResponseReasoningItem.model_validate(added["reasoning"])
    assert added["reasoning"]["summary"] == []
    ResponseOutputMessage.model_validate(added["message"])
    assert added["message"]["content"] == []
    call = ResponseFunctionToolCall.model_validate(added["function_call"])
    assert (call.call_id, call.name, call.arguments) == ("call_1", "lookup", "")

    prefilled = {**added["function_call"]}
    del prefilled["call_id"]
    with pytest.raises(ValidationError):
        ResponseFunctionToolCall.model_validate(prefilled)


def test_added_items_never_prefill_the_content_their_done_events_carry() -> None:
    projected = _projected_stream()
    added = {
        item["item"]["id"]: item["item"]
        for item in projected
        if item["type"] == "response.output_item.added"
    }
    done = {
        item["item"]["id"]: item["item"]
        for item in projected
        if item["type"] == "response.output_item.done"
    }
    assert set(added) == set(done)
    assert done["msg_1"]["content"][0]["text"] == "final answer"
    assert added["msg_1"]["content"] == []
    assert done["rs_1"]["summary"][0]["text"] == "chain thought"
    assert added["rs_1"]["summary"] == []
    assert done["fc_1"]["arguments"] == '{"q":"cats"}'
    assert added["fc_1"]["arguments"] == ""


def test_public_alias_is_the_only_model_on_the_wire() -> None:
    """One public value, one internal value, never conflated."""

    events = _mixed_lifecycle_events()
    started = events[0]
    assert isinstance(started, TurnStarted)
    assert started.model == _PHYSICAL_MODEL
    assert started.requested_model == _PUBLIC_ALIAS

    projected = _projected_stream(events)
    models = {
        item["response"]["model"]
        for item in projected
        if item["type"] in _LIFECYCLE_EVENT_TYPES
    }
    assert models == {_PUBLIC_ALIAS}
    assert _PHYSICAL_MODEL not in json.dumps(projected)


def test_snapshot_identity_and_settings_are_stable_across_the_whole_stream() -> None:
    projected = _projected_stream()
    lifecycle = tuple(
        item["response"] for item in projected if item["type"] in _LIFECYCLE_EVENT_TYPES
    )
    terminal = lifecycle[-1]
    for snapshot in lifecycle:
        for field in ("id", "created_at", "model", "tools", "tool_choice"):
            assert snapshot[field] == terminal[field], field
        assert snapshot["object"] == "response"

    sequence_numbers = [item["sequence_number"] for item in projected]
    assert sequence_numbers == sorted(sequence_numbers)
    assert len(set(sequence_numbers)) == len(sequence_numbers)


def test_terminal_snapshot_reconstructs_every_output_item_exactly_once() -> None:
    terminal = next(
        item for item in _projected_stream() if item["type"] == "response.completed"
    )["response"]
    assert [item["id"] for item in terminal["output"]] == ["rs_1", "msg_1", "fc_1"]
    assert terminal["output"][2]["call_id"] == "call_1"
    assert terminal["usage"]["total_tokens"] == 24


def test_reasoning_and_visible_text_never_cross_channels() -> None:
    projected = _projected_stream()
    reasoning_text = "".join(
        item["delta"]
        for item in projected
        if item["type"] == "response.reasoning_summary_text.delta"
    )
    visible_text = "".join(
        item["delta"]
        for item in projected
        if item["type"] == "response.output_text.delta"
    )
    assert reasoning_text == "chain thought"
    assert visible_text == "final answer"
    assert reasoning_text not in visible_text
    for item in projected:
        if item["type"].startswith("response.output_text"):
            assert "chain thought" not in json.dumps(item)
        if item["type"].startswith("response.reasoning_summary_text"):
            assert "final answer" not in json.dumps(item)
