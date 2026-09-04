"""RED contract for the real 3more multimodal Responses workload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_batch_server.responses.transport import (
    ResponseCreateCommand,
    ResponseSteerCommand,
    StreamId,
    TransportProtocolError,
    parse_websocket_command,
)

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "three_more_north_star.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _commands() -> dict[str, dict[str, object]]:
    commands = _fixture()["commands"]
    assert isinstance(commands, dict)
    return commands


def test_three_more_fixture_is_pinned_to_the_mapped_consumer() -> None:
    fixture = _fixture()
    consumer = fixture["consumer"]
    assert isinstance(consumer, dict)

    assert consumer["repository"] == ("https://github.com/div0-space/3more-studio.git")
    assert consumer["commit"] == ("c20c31e1789ad82612d719c307dee96d67f27853")
    assert consumer["current_transport"] == "responses-sse"
    assert consumer["target_transport"] == "responses-websocket"
    assert consumer["fallback_transport"] == "responses-sse"
    assert consumer["max_images_per_turn"] == 8
    assert consumer["max_tool_rounds"] == 8


def test_three_more_initial_round_is_lossless_and_parseable() -> None:
    command = parse_websocket_command(_commands()["photo_initial"])

    assert isinstance(command, ResponseCreateCommand)
    assert command.stream_id == StreamId("studio.photo")
    body = command.response
    assert body["store"] is False
    assert body["parallel_tool_calls"] is True
    assert "stream" not in body
    assert "background" not in body

    input_items = body["input"]
    assert isinstance(input_items, list)
    message = input_items[0]
    assert isinstance(message, dict)
    content = message["content"]
    assert isinstance(content, list)
    assert sum(part["type"] == "input_image" for part in content) == 8
    assert sum(part["type"] == "input_file" for part in content) == 1
    file_part = next(part for part in content if part["type"] == "input_file")
    assert file_part == {
        "type": "input_file",
        "file_data": "${BRIEF_FILE_DATA}",
        "filename": "session-brief.pdf",
    }


def test_three_more_tool_continuation_preserves_receipt_before_view() -> None:
    command = parse_websocket_command(_commands()["photo_tool_continuation"])

    assert isinstance(command, ResponseCreateCommand)
    assert command.stream_id == StreamId("studio.photo")
    assert command.response["previous_response_id"] == "${PHOTO_RESPONSE_ID}"
    assert "instructions" not in command.response
    items = command.response["input"]
    assert isinstance(items, list)
    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == "${INSPECT_CANVAS_CALL_ID}"
    assert items[1]["type"] == "message"
    assert items[1]["content"][1]["type"] == "input_image"


def test_three_more_parallel_lenses_use_distinct_ordered_lanes() -> None:
    photo = parse_websocket_command(_commands()["photo_initial"])
    writer = parse_websocket_command(_commands()["writer_parallel"])

    assert isinstance(photo, ResponseCreateCommand)
    assert isinstance(writer, ResponseCreateCommand)
    assert photo.stream_id == StreamId("studio.photo")
    assert writer.stream_id == StreamId("studio.writer")
    assert photo.stream_id != writer.stream_id


def test_three_more_multimodal_steer_shape_is_capability_gated() -> None:
    connection = _fixture()["connection_contract"]
    assert isinstance(connection, dict)
    assert connection["routine_client_events"] == ["response.create"]
    assert connection["capability_gated_client_events"] == ["response.steer"]
    assert "steering_not_supported" in connection["steer_policy"]

    command = parse_websocket_command(_commands()["photo_steer"])

    assert isinstance(command, ResponseSteerCommand)
    assert command.previous_response_id == "${ACTIVE_PHOTO_RESPONSE_ID}"
    assert isinstance(command.input, tuple)
    content = command.input[0]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == [
        "input_text",
        "input_image",
    ]


@pytest.mark.parametrize(
    "event_type",
    ["response.cancel", "ping"],
)
def test_three_more_routine_websocket_rejects_custom_events(
    event_type: str,
) -> None:
    with pytest.raises(TransportProtocolError):
        parse_websocket_command({"type": event_type})


def test_three_more_fixture_names_full_acceptance_surface() -> None:
    fixture = _fixture()
    invariants = fixture["acceptance_invariants"]
    assert isinstance(invariants, list)

    required_fragments = (
        "eight_images",
        "reasoning_and_output_text",
        "side_effect_executes_once",
        "receipts_before_canvas_views",
        "previous_response_id",
        "stream_id",
        "terminal_sse_and_websocket",
        "runtime_cleanup",
        "beta_response_inject",
        "flash_steering",
        "buddy_quality",
        "no_cloud_fallback",
    )
    assert all(
        any(fragment in invariant for invariant in invariants)
        for fragment in required_fragments
    )
