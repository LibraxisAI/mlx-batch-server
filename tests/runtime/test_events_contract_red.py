"""RED contracts for the neutral hosted event family (design §1.4)."""

from __future__ import annotations

import dataclasses

import pytest

from mlx_batch_server.runtime.events import (
    HOSTED_CALL_ITEM_KIND,
    OUTPUT_ITEM_KINDS,
    TURN_EVENT_TYPES,
    HostedCallCompleted,
    HostedCallProgress,
    HostedCallStarted,
    OutputItemCompleted,
    OutputItemStarted,
)


def test_hosted_call_is_a_first_class_output_item_kind() -> None:
    assert HOSTED_CALL_ITEM_KIND == "hosted_call"
    assert HOSTED_CALL_ITEM_KIND in OUTPUT_ITEM_KINDS
    assert "function_call" in OUTPUT_ITEM_KINDS


def test_hosted_events_are_members_of_the_closed_turn_event_union() -> None:
    for event_type in (HostedCallStarted, HostedCallProgress, HostedCallCompleted):
        assert event_type in TURN_EVENT_TYPES


def test_hosted_call_started_freezes_its_action() -> None:
    event = HostedCallStarted(
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        tool_name="web_search",
        action={"query": "loctree", "nested": {"k": [1, 2]}},
    )
    with pytest.raises(TypeError):
        event.action["query"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        event.action["nested"]["k"] = []  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.call_id = "call_2"  # type: ignore[misc]


def test_hosted_call_completed_receipt_is_deep_frozen() -> None:
    event = HostedCallCompleted(
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        tool_name="web_fetch",
        status="failed",
        receipt={"call_id": "call_1", "error": {"code": "tool_timeout"}},
    )
    with pytest.raises(TypeError):
        event.receipt["status"] = "completed"  # type: ignore[index]
    with pytest.raises(TypeError):
        event.receipt["error"]["code"] = "other"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.status = "completed"  # type: ignore[misc]


def test_hosted_call_status_is_a_closed_set() -> None:
    with pytest.raises(ValueError):
        HostedCallCompleted(
            index=0,
            item_id="hosted_1",
            call_id="call_1",
            tool_name="web_search",
            status="in_progress",
        )


def test_hosted_events_reject_empty_identity() -> None:
    with pytest.raises(ValueError):
        HostedCallStarted(index=0, item_id="", call_id="c", tool_name="t")
    with pytest.raises(ValueError):
        HostedCallProgress(index=0, item_id="i", call_id="c", phase=" ")
    with pytest.raises(ValueError):
        HostedCallCompleted(
            index=-1,
            item_id="i",
            call_id="c",
            tool_name="t",
            status="completed",
        )


def test_hosted_call_output_item_start_requires_tool_identity() -> None:
    with pytest.raises(ValueError):
        OutputItemStarted(kind=HOSTED_CALL_ITEM_KIND, index=0, item_id="hosted_1")
    event = OutputItemStarted(
        kind=HOSTED_CALL_ITEM_KIND,
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        name="web_search",
    )
    assert event.call_id == "call_1"


def test_hosted_call_output_item_completion_permits_failed_status() -> None:
    event = OutputItemCompleted(
        kind=HOSTED_CALL_ITEM_KIND,
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        name="web_search",
        status="failed",
    )
    assert event.status == "failed"
    with pytest.raises(ValueError):
        OutputItemCompleted(
            kind=HOSTED_CALL_ITEM_KIND,
            index=0,
            item_id="hosted_1",
            call_id="call_1",
            name="web_search",
            status="incomplete",
        )
    with pytest.raises(ValueError):
        OutputItemCompleted(
            kind=HOSTED_CALL_ITEM_KIND,
            index=0,
            item_id="hosted_1",
            call_id="call_1",
            name="web_search",
            text="no text allowed",
        )
    with pytest.raises(ValueError):
        # message items must not gain the failed hosted status.
        OutputItemCompleted(
            kind="message",
            index=0,
            item_id="msg_1",
            text="hello",
            status="failed",
        )
