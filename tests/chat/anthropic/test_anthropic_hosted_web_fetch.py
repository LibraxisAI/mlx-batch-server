"""Executable Anthropic 0.96.0 oracle for native hosted web fetch."""

from __future__ import annotations

from typing import Any

import anthropic
import pytest
from fastapi.testclient import TestClient

from mlx_batch_server.chat.anthropic.anthropic_schema import MessagesRequest
from mlx_batch_server.chat.anthropic.capabilities import (
    detached_profile,
    enforce_capabilities,
)
from mlx_batch_server.chat.anthropic.errors import (
    AnthropicAPIError,
    UnsupportedCapabilityError,
)
from mlx_batch_server.chat.anthropic.projector import AnthropicMessageProjector
from mlx_batch_server.chat.anthropic.request_mapper import build_turn
from mlx_batch_server.chat.anthropic.turn_source import (
    AnthropicTurn,
    clear_turn_source,
    register_turn_source,
)
from mlx_batch_server.main import app
from mlx_batch_server.runtime.agentic import CITATIONS_METADATA_KEY
from mlx_batch_server.runtime.events import (
    TEXT_CONTENT_KIND,
    ContentPartStarted,
    HostedCallCompleted,
    HostedCallResult,
    HostedCallStarted,
    HostedCitation,
    TextCompleted,
    TextDelta,
    TurnCancelled,
    TurnCompleted,
    TurnStarted,
    UsageUpdate,
)

_URL = "https://example.test/final"
_SOURCE = "Alpha grounded omega."
_CONTINUATION = "The source says grounded."
_SOURCE_START = _SOURCE.index("grounded")
_OUTPUT_START = _CONTINUATION.index("grounded")
_RESULT = {
    "kind": "document",
    "url": _URL,
    "media_type": "text/plain; charset=utf-8",
    "content": _SOURCE,
    "digest": "sha256:" + "a" * 64,
    "retrieved_at": 1_757_088_000,
}


def _request(**tool_overrides: Any) -> MessagesRequest:
    tool = {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 2,
        "max_content_tokens": 4096,
        "allowed_domains": ["EXAMPLE.test.", "example.test"],
        "citations": {"enabled": True},
    }
    tool.update(tool_overrides)
    return MessagesRequest.model_validate(
        {
            "model": "qwen-flash",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Fetch the page"}],
            "tools": [tool],
        }
    )


def _citation(**overrides: Any) -> HostedCitation:
    values: dict[str, Any] = {
        "output_index": 1,
        "item_id": "text_1",
        "content_index": 0,
        "source_call_id": "call_fetch",
        "source_url": _URL,
        "cited_text": "grounded",
        "source_start": _SOURCE_START,
        "source_end": _SOURCE_START + len("grounded"),
        "output_start": _OUTPUT_START,
        "output_end": _OUTPUT_START + len("grounded"),
    }
    values.update(overrides)
    return HostedCitation(**values)


def _success_events(*, include_citation: bool = True) -> list[Any]:
    events = [
        TurnStarted(response_id="resp_fetch", model="physical", created_at=1),
        HostedCallStarted(
            0,
            "hosted_fetch",
            "call_fetch",
            "web_fetch",
            {"url": "https://example.test/start"},
        ),
        HostedCallResult(
            0,
            "hosted_fetch",
            "call_fetch",
            "web_fetch",
            _RESULT,
        ),
        HostedCallCompleted(
            0,
            "hosted_fetch",
            "call_fetch",
            "web_fetch",
            "completed",
            {
                "status": "completed",
                "final_url": _URL,
                "result_digest": _RESULT["digest"],
            },
        ),
        ContentPartStarted(TEXT_CONTENT_KIND, 1, 0, "text_1"),
        TextDelta(_CONTINUATION, "text_1", 1, 0),
    ]
    if include_citation:
        events.append(_citation())
    events.extend(
        [
            TextCompleted(_CONTINUATION, "text_1", 1, 0),
            TurnCompleted(
                "stop",
                usage=UsageUpdate(input_tokens=7, output_tokens=5, total_tokens=12),
            ),
        ]
    )
    return events


def _citation_ready_projector(
    *,
    result: dict[str, Any] | None = None,
    citations_enabled: bool = False,
) -> AnthropicMessageProjector:
    stored_result = _RESULT if result is None else result
    projector = AnthropicMessageProjector(
        message_id="msg_citation",
        model_alias="m",
        citations_enabled=citations_enabled,
    )
    projector.observe(
        HostedCallStarted(
            0,
            "hosted_fetch",
            "call_fetch",
            "web_fetch",
            {"url": "https://example.test/start"},
        )
    )
    projector.observe(
        HostedCallResult(
            0,
            "hosted_fetch",
            "call_fetch",
            "web_fetch",
            stored_result,
        )
    )
    projector.observe(
        HostedCallCompleted(
            0,
            "hosted_fetch",
            "call_fetch",
            "web_fetch",
            "completed",
            {"final_url": _URL, "result_digest": stored_result["digest"]},
        )
    )
    projector.observe(ContentPartStarted(TEXT_CONTENT_KIND, 1, 0, "text_1"))
    projector.observe(TextDelta(_CONTINUATION, "text_1", 1, 0))
    return projector


class _ScriptedSource:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.turns: list[AnthropicTurn] = []

    def stream(self, turn: AnthropicTurn):
        self.turns.append(turn)

        async def generate():
            for event in self.events:
                yield event

        return generate()


def test_web_fetch_request_is_normalized_onto_the_neutral_hosted_descriptor() -> None:
    request = _request()
    admission = enforce_capabilities(request, detached_profile(request.model))
    turn = build_turn(request)

    assert {entry.key for entry in admission.normalized} >= {
        "tools.web_fetch.max_uses",
        "tools.web_fetch.allowed_domains",
        "tools.web_fetch.max_content_tokens",
    }
    assert turn.tools == (
        {
            "type": "web_fetch",
            "name": "web_fetch",
            "max_uses": 2,
            "max_content_tokens": 4096,
            "allowed_domains": ("example.test",),
            "citations": {"enabled": True},
        },
    )
    assert turn.metadata[CITATIONS_METADATA_KEY] is True


@pytest.mark.parametrize(
    ("tools", "path"),
    [
        (
            [{"type": "web_fetch_20260209", "name": "web_fetch"}],
            "tools.0.type",
        ),
        (
            [
                {"type": "web_fetch_20250910", "name": "web_fetch"},
                {
                    "name": "client_tool",
                    "input_schema": {"type": "object"},
                },
            ],
            "tools.1",
        ),
        (
            [{"type": "web_search_20250305", "name": "web_search"}],
            "tools.0.type",
        ),
    ],
)
def test_unsupported_hosted_forms_fail_before_inference(
    tools: list[dict[str, Any]], path: str
) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "qwen-flash",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": tools,
        }
    )
    with pytest.raises(UnsupportedCapabilityError, match=path.replace(".", r"\.")):
        admission = enforce_capabilities(request, detached_profile(request.model))
        build_turn(request)
        raise AssertionError(admission)


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_mixed_tools_fail_as_anthropic_400_before_source_entry(stream: bool) -> None:
    source = _ScriptedSource([])
    register_turn_source(source)
    try:
        response = TestClient(app).post(
            "/anthropic/v1/messages",
            json={
                "model": "qwen-flash",
                "max_tokens": 16,
                "stream": stream,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {"type": "web_fetch_20250910", "name": "web_fetch"},
                    {
                        "name": "client_tool",
                        "input_schema": {"type": "object"},
                    },
                ],
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert "tools.1" in response.json()["error"]["message"]
        assert not source.turns
        assert "text/event-stream" not in response.headers.get("content-type", "")
    finally:
        clear_turn_source(source)


def test_mutually_exclusive_domain_filters_are_rejected_at_the_field() -> None:
    request = _request(blocked_domains=["blocked.test"])
    with pytest.raises(AnthropicAPIError, match="mutually exclusive"):
        build_turn(request)


def test_duplicate_result_unknown_citation_and_pdf_success_fail_closed() -> None:
    projector = AnthropicMessageProjector(
        message_id="msg_mutation", model_alias="m", citations_enabled=True
    )
    started = HostedCallStarted(
        0, "hosted_fetch", "call_fetch", "web_fetch", {"url": _URL}
    )
    result = HostedCallResult(0, "hosted_fetch", "call_fetch", "web_fetch", _RESULT)
    projector.observe(started)
    projector.observe(result)
    with pytest.raises(AnthropicAPIError, match="more than one result"):
        projector.observe(result)

    unknown = HostedCitation(
        output_index=1,
        item_id="text_1",
        content_index=0,
        source_call_id="unknown_call",
        source_url=_URL,
        cited_text="grounded",
        source_start=0,
        source_end=8,
        output_start=0,
        output_end=8,
    )
    with pytest.raises(AnthropicAPIError, match="no completed web_fetch receipt"):
        projector.observe(unknown)

    pdf = AnthropicMessageProjector(message_id="msg_pdf", model_alias="m")
    pdf.observe(started)
    pdf_result = dict(_RESULT, media_type="application/pdf")
    pdf.observe(
        HostedCallResult(0, "hosted_fetch", "call_fetch", "web_fetch", pdf_result)
    )
    with pytest.raises(AnthropicAPIError, match="not text-family"):
        pdf.observe(
            HostedCallCompleted(
                0,
                "hosted_fetch",
                "call_fetch",
                "web_fetch",
                "completed",
                {"final_url": _URL, "result_digest": _RESULT["digest"]},
            )
        )


@pytest.mark.parametrize(
    ("result", "citation"),
    [
        pytest.param(
            _RESULT,
            _citation(source_start=0, source_end=len("grounded")),
            id="mutated-source-range",
        ),
        pytest.param(
            dict(_RESULT, content="Alpha tampered omega."),
            _citation(),
            id="mutated-source-text",
        ),
    ],
)
def test_citation_requires_exact_fetched_source_slice(
    result: dict[str, Any], citation: HostedCitation
) -> None:
    projector = _citation_ready_projector(
        result=result,
        citations_enabled=True,
    )

    with pytest.raises(AnthropicAPIError, match="not verbatim web_fetch document"):
        projector.observe(citation)


def test_citation_event_fails_closed_when_request_did_not_enable_it() -> None:
    projector = _citation_ready_projector()

    with pytest.raises(AnthropicAPIError, match="citations were not enabled"):
        projector.observe(_citation())


def test_cancel_is_a_hard_barrier_against_late_result_and_continuation() -> None:
    projector = AnthropicMessageProjector(message_id="msg_cancel", model_alias="m")
    projector.observe(
        HostedCallStarted(0, "hosted_fetch", "call_fetch", "web_fetch", {"url": _URL})
    )
    projected = projector.observe(TurnCancelled("client_disconnected"))
    assert [event.type for event in projected] == ["error"]

    assert (
        projector.observe(
            HostedCallCompleted(
                0,
                "hosted_fetch",
                "call_fetch",
                "web_fetch",
                "failed",
                {"error": {"code": "fetch_url_fetch_cancelled", "message": "cancel"}},
            )
        )
        == ()
    )
    assert projector.observe(TextDelta("hidden", "text_1", 1, 0)) == ()
    assert [block.type for block in projector.content_blocks()] == ["server_tool_use"]


def test_sdk_parses_unary_and_streamed_success_with_the_same_blocks() -> None:
    source = _ScriptedSource(_success_events())
    register_turn_source(source)
    try:
        client = anthropic.Anthropic(
            base_url="http://test/anthropic",
            api_key="not-needed",
            http_client=TestClient(app),
        )
        kwargs = {
            "model": "qwen-flash",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Fetch the page"}],
            "tools": [
                {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "citations": {"enabled": True},
                }
            ],
        }
        unary = client.messages.create(**kwargs)
        with client.messages.stream(**kwargs) as stream:
            streamed_events = list(stream)
            streamed = stream.get_final_message()

        assert [block.type for block in unary.content] == [
            "server_tool_use",
            "web_fetch_tool_result",
            "text",
        ]
        assert [block.type for block in streamed.content] == [
            "server_tool_use",
            "web_fetch_tool_result",
            "text",
        ]
        assert streamed.content[0].id == unary.content[0].id
        assert streamed.content[0].input == unary.content[0].input
        assert streamed.content[1].tool_use_id == unary.content[1].tool_use_id
        # SDK 0.96.0 parses the start event as WebFetchToolResultBlock but its
        # stream accumulator preserves the nested content as a dict.
        assert streamed.content[1].content["url"] == unary.content[1].content.url
        assert streamed.content[2].text == unary.content[2].text
        assert streamed.content[2].citations == unary.content[2].citations
        assert streamed.usage == unary.usage
        assert streamed.stop_reason == unary.stop_reason
        server_use, result, text = unary.content
        assert server_use.id.startswith("srvtoolu_")
        assert server_use.input == {"url": "https://example.test/start"}
        assert result.tool_use_id == server_use.id
        assert result.content.url == _URL
        assert result.content.content.source.data == _SOURCE
        document_citations = result.content.content.citations
        assert document_citations is not None
        assert document_citations.model_dump() == {"enabled": True}
        assert streamed.content[1].content["content"]["citations"] == {"enabled": True}
        assert text.citations[0].start_char_index == _SOURCE_START
        assert text.citations[0].end_char_index == _SOURCE_START + len("grounded")
        assert unary.usage.input_tokens == 7
        assert unary.usage.output_tokens == 5
        assert (
            sum(event.type == "content_block_start" for event in streamed_events) == 3
        )
        assert not any(
            getattr(getattr(event, "delta", None), "type", None)
            == "web_fetch_result_delta"
            for event in streamed_events
        )
    finally:
        clear_turn_source(source)


@pytest.mark.parametrize(
    "citations",
    [None, {"enabled": False}],
    ids=["omitted", "false"],
)
def test_sdk_does_not_widen_disabled_or_omitted_citations(
    citations: dict[str, bool] | None,
) -> None:
    source = _ScriptedSource(_success_events(include_citation=False))
    register_turn_source(source)
    try:
        client = anthropic.Anthropic(
            base_url="http://test/anthropic",
            api_key="not-needed",
            http_client=TestClient(app),
        )
        tool: dict[str, Any] = {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
        }
        if citations is not None:
            tool["citations"] = citations
        kwargs = {
            "model": "qwen-flash",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Fetch the page"}],
            "tools": [tool],
        }

        unary = client.messages.create(**kwargs)
        with client.messages.stream(**kwargs) as stream:
            streamed = stream.get_final_message()

        assert unary.content[1].content.content.citations is None
        assert streamed.content[1].content["content"].get("citations") is None
        assert unary.content[2].citations is None
        assert streamed.content[2].citations is None
    finally:
        clear_turn_source(source)


def test_sdk_parses_typed_fetch_failure_and_one_continuation() -> None:
    events = [
        TurnStarted(response_id="resp_fetch_error", model="physical", created_at=1),
        HostedCallStarted(
            0,
            "hosted_fetch",
            "call_fetch",
            "web_fetch",
            {"url": "http://127.0.0.1/private"},
        ),
        HostedCallCompleted(
            0,
            "hosted_fetch",
            "call_fetch",
            "web_fetch",
            "failed",
            {"error": {"code": "fetch_url_target_blocked", "message": "blocked"}},
        ),
        ContentPartStarted(TEXT_CONTENT_KIND, 1, 0, "text_1"),
        TextDelta("The fetch was blocked by policy.", "text_1", 1, 0),
        TextCompleted("The fetch was blocked by policy.", "text_1", 1, 0),
        TurnCompleted(
            "stop",
            usage=UsageUpdate(input_tokens=5, output_tokens=6, total_tokens=11),
        ),
    ]
    source = _ScriptedSource(events)
    register_turn_source(source)
    try:
        client = anthropic.Anthropic(
            base_url="http://test/anthropic",
            api_key="not-needed",
            http_client=TestClient(app),
        )
        kwargs = {
            "model": "qwen-flash",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Fetch private"}],
            "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}],
        }
        response = client.messages.create(**kwargs)
        with client.messages.stream(**kwargs) as stream:
            streamed = stream.get_final_message()
        assert [block.type for block in response.content] == [
            "server_tool_use",
            "web_fetch_tool_result",
            "text",
        ]
        assert response.content[1].content.type == "web_fetch_tool_result_error"
        assert response.content[1].content.error_code == "url_not_allowed"
        assert response.content[2].text == "The fetch was blocked by policy."
        assert [block.type for block in streamed.content] == [
            "server_tool_use",
            "web_fetch_tool_result",
            "text",
        ]
        assert streamed.content[1].content["error_code"] == "url_not_allowed"
        assert streamed.content[2].text == response.content[2].text
        assert streamed.usage == response.usage
        assert streamed.stop_reason == response.stop_reason
    finally:
        clear_turn_source(source)
