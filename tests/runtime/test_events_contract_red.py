"""RED contracts for the neutral hosted event family (design §1.4)."""

from __future__ import annotations

import dataclasses

import pytest

from mlx_batch_server.runtime.events import (
    HOSTED_CALL_ITEM_KIND,
    MAX_CITED_TEXT_CHARS,
    OUTPUT_ITEM_KINDS,
    TURN_EVENT_TYPES,
    HostedCallCompleted,
    HostedCallProgress,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
    OutputItemCompleted,
    OutputItemStarted,
)


def _sealed_search(**overrides: object) -> dict[str, object]:
    action: dict[str, object] = {
        "kind": "search",
        "query": "loctree",
        "sources": ["https://example.com/a"],
    }
    action.update(overrides)
    return action


def test_hosted_call_is_a_first_class_output_item_kind() -> None:
    assert HOSTED_CALL_ITEM_KIND == "hosted_call"
    assert HOSTED_CALL_ITEM_KIND in OUTPUT_ITEM_KINDS
    assert "function_call" in OUTPUT_ITEM_KINDS


def test_hosted_events_are_members_of_the_closed_turn_event_union() -> None:
    for event_type in (
        HostedCallStarted,
        HostedCallProgress,
        HostedCallResult,
        HostedCitation,
        HostedCallCompleted,
    ):
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


def test_hosted_call_output_item_start_requires_identity_and_opening_action() -> None:
    with pytest.raises(ValueError):
        OutputItemStarted(kind=HOSTED_CALL_ITEM_KIND, index=0, item_id="hosted_1")
    with pytest.raises(ValueError, match="requires its opening action"):
        OutputItemStarted(
            kind=HOSTED_CALL_ITEM_KIND,
            index=0,
            item_id="hosted_1",
            call_id="call_1",
            name="web_search",
        )
    with pytest.raises(ValueError, match="non-empty action"):
        OutputItemStarted(
            kind=HOSTED_CALL_ITEM_KIND,
            index=0,
            item_id="hosted_1",
            call_id="call_1",
            name="web_search",
            action={},
        )
    event = OutputItemStarted(
        kind=HOSTED_CALL_ITEM_KIND,
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        name="web_search",
        action={"query": "loctree", "nested": {"values": [1, 2]}},
    )
    assert event.call_id == "call_1"
    assert event.action is not None
    assert event.action["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        event.action["query"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError):
        event.action["nested"]["values"] = ()  # type: ignore[index]


def test_only_hosted_item_starts_may_carry_an_action() -> None:
    for kind, identity in (
        ("message", {}),
        ("reasoning", {}),
        ("function_call", {"call_id": "call_1", "name": "lookup"}),
    ):
        with pytest.raises(ValueError, match="cannot carry a hosted action"):
            OutputItemStarted(
                kind=kind,
                index=0,
                item_id=f"{kind}_1",
                action={"query": "loctree"},
                **identity,  # type: ignore[arg-type]
            )


def test_hosted_call_output_item_completion_permits_failed_status() -> None:
    event = OutputItemCompleted(
        kind=HOSTED_CALL_ITEM_KIND,
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        name="web_search",
        status="failed",
        action=_sealed_search(sources=[]),
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
            action=_sealed_search(),
        )
    with pytest.raises(ValueError):
        OutputItemCompleted(
            kind=HOSTED_CALL_ITEM_KIND,
            index=0,
            item_id="hosted_1",
            call_id="call_1",
            name="web_search",
            text="no text allowed",
            action=_sealed_search(),
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


def test_hosted_completion_action_is_required_exactly_for_hosted_items() -> None:
    with pytest.raises(ValueError, match="requires its sealed action"):
        OutputItemCompleted(
            kind=HOSTED_CALL_ITEM_KIND,
            index=0,
            item_id="hosted_1",
            call_id="call_1",
            name="web_search",
        )
    for kind, payload in (
        ("message", {"text": "hello"}),
        ("reasoning", {"text": "because"}),
        (
            "function_call",
            {"call_id": "call_1", "name": "search", "arguments": "{}"},
        ),
    ):
        with pytest.raises(ValueError, match="cannot carry a hosted action"):
            OutputItemCompleted(
                kind=kind,
                index=0,
                item_id=f"{kind}_1",
                action={"kind": "fetch", "url": "https://example.com"},
                **payload,  # type: ignore[arg-type]
            )


def test_hosted_completion_action_is_a_closed_deep_frozen_schema() -> None:
    event = OutputItemCompleted(
        kind=HOSTED_CALL_ITEM_KIND,
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        name="web_search",
        action=_sealed_search(),
    )
    assert event.action is not None
    assert event.action["sources"] == ("https://example.com/a",)
    with pytest.raises(TypeError):
        event.action["query"] = "mutated"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.action = None  # type: ignore[misc]

    hosted = {
        "kind": HOSTED_CALL_ITEM_KIND,
        "index": 0,
        "item_id": "hosted_1",
        "call_id": "call_1",
        "name": "web_search",
    }
    with pytest.raises(ValueError, match="kind must be search or fetch"):
        OutputItemCompleted(**hosted, action={"kind": "browse"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly kind, query and sources"):
        OutputItemCompleted(**hosted, action=_sealed_search(extra="x"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="action query"):
        OutputItemCompleted(**hosted, action=_sealed_search(query="  "))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be unique"):
        OutputItemCompleted(
            **hosted,  # type: ignore[arg-type]
            action=_sealed_search(sources=["https://a", "https://a"]),
        )
    with pytest.raises(ValueError, match="exactly kind and url"):
        OutputItemCompleted(**hosted, action={"kind": "fetch"})  # type: ignore[arg-type]


def test_failed_hosted_action_carries_no_success_sources() -> None:
    with pytest.raises(ValueError, match="cannot carry success sources"):
        OutputItemCompleted(
            kind=HOSTED_CALL_ITEM_KIND,
            index=0,
            item_id="hosted_1",
            call_id="call_1",
            name="web_search",
            status="failed",
            action=_sealed_search(),
        )


def test_hosted_result_validates_identity_and_freezes_its_payload() -> None:
    event = HostedCallResult(
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        tool_name="web_fetch",
        result={
            "kind": "document",
            "url": "https://example.com/doc",
            "media_type": "text/plain",
            "content": "fetched body",
            "digest": "sha256:abc",
            "retrieved_at": 1757000000,
        },
    )
    assert event.identities == ("https://example.com/doc",)
    with pytest.raises(TypeError):
        event.result["content"] = "mutated"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.call_id = "call_2"  # type: ignore[misc]

    search = HostedCallResult(
        index=0,
        item_id="hosted_1",
        call_id="call_1",
        tool_name="web_search",
        result={
            "kind": "search_results",
            "query": "loctree",
            "results": [
                {"title": "a", "url": "https://a", "snippet": "s"},
                {"title": "b", "url": "https://b", "snippet": "s"},
            ],
            "digest": "sha256:def",
        },
    )
    assert search.identities == ("https://a", "https://b")

    base = {
        "index": 0,
        "item_id": "hosted_1",
        "call_id": "call_1",
        "tool_name": "web_fetch",
    }
    with pytest.raises(ValueError, match="document or search_results"):
        HostedCallResult(**base, result={"url": "https://a", "digest": "sha256:x"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="result digest"):
        HostedCallResult(**base, result={"kind": "document", "url": "https://a"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="result url"):
        HostedCallResult(**base, result={"kind": "document", "digest": "sha256:x"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="results sequence"):
        HostedCallResult(
            **base,  # type: ignore[arg-type]
            result={"kind": "search_results", "digest": "sha256:x", "results": "no"},
        )
    with pytest.raises(ValueError, match="search result url"):
        HostedCallResult(
            **base,  # type: ignore[arg-type]
            result={
                "kind": "search_results",
                "digest": "sha256:x",
                "results": [{"title": "a"}],
            },
        )
    with pytest.raises(ValueError, match="item_id"):
        HostedCallResult(
            index=0,
            item_id=" ",
            call_id="call_1",
            tool_name="web_fetch",
            result={"kind": "document", "url": "https://a", "digest": "sha256:x"},
        )


def _citation(**overrides: object) -> HostedCitation:
    payload: dict[str, object] = {
        "output_index": 1,
        "item_id": "message_1",
        "content_index": 0,
        "source_call_id": "call_1",
        "source_url": "https://example.com/doc",
        "cited_text": "quoted",
        "source_start": 10,
        "source_end": 16,
        "output_start": 0,
        "output_end": 6,
    }
    payload.update(overrides)
    return HostedCitation(**payload)  # type: ignore[arg-type]


def test_hosted_citation_enforces_identity_range_and_text_bounds() -> None:
    citation = _citation()
    assert citation.cited_text == "quoted"
    with pytest.raises(dataclasses.FrozenInstanceError):
        citation.source_url = "https://other"  # type: ignore[misc]

    with pytest.raises(ValueError, match="source_call_id"):
        _citation(source_call_id=" ")
    with pytest.raises(ValueError, match="source_url"):
        _citation(source_url="")
    with pytest.raises(ValueError, match="cited_text must not be empty"):
        _citation(cited_text="")
    with pytest.raises(ValueError, match=f"{MAX_CITED_TEXT_CHARS}"):
        _citation(cited_text="x" * (MAX_CITED_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="source_end must exceed"):
        _citation(source_start=16, source_end=16)
    with pytest.raises(ValueError, match="output_end must exceed"):
        _citation(output_start=6, output_end=3)
    with pytest.raises(ValueError, match="source_start"):
        _citation(source_start=-1)
